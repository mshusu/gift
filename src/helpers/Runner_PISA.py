import os
import gc
import copy
import torch
import logging
import hashlib
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Dataloader
from utils import utils
from models.Model import Model
import Inference
import json

class Runner_PISA:
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=200, help='Number of epochs for fine-tuning.')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04, help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256, help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: Adam')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of processors for DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1, help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='', help='Path for test results')
        parser.add_argument('--pisa_debug_parity', type=int, default=0,
                            help='Print PISA parity debug information for loader mode, batch losses, and validation.')
        parser.add_argument('--pisa_aux_optimizer_mode', type=str, default='reuse',
                            choices=['reuse', 'reset', 'load_forward'],
                            help='Optimizer state used for the PISA auxiliary step: reuse current, reset, or load forward checkpoint optimizer.')
        parser.add_argument('--pisa_kmeans_seed_mode', type=str, default='global',
                            choices=['global', 'epoch', 'isolated_epoch'],
                            help='PISA k-means RNG mode. global keeps existing behavior; epoch reseeds before k-means; isolated_epoch reseeds only inside k-means.')
        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = getattr(args, 'persistent_workers', 1)
        self.prefetch_factor = getattr(args, 'prefetch_factor', 4)
        self.shuffle = bool(getattr(args, 'shuffle', 1))
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch
        self.pisa_debug_parity = bool(getattr(args, 'pisa_debug_parity', 0))
        self.random_seed = getattr(args, 'random_seed', 0)
        self.pisa_kmeans_seed_mode = getattr(args, 'pisa_kmeans_seed_mode', 'global')
        self.time = None
        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path

    def _debug_parity(self, msg):
        if self.pisa_debug_parity:
            print(msg, flush=True)
            logging.info(msg)

    def _effective_legacy_neg_sampler(self, args):
        legacy_flag = getattr(args, 'legacy_aux_neg_sampler', -1)
        legacy_models = {'PISA_LGN', 'Contrastive_LGN'}
        if legacy_flag < 0:
            return int(getattr(args, 'model_name', '') in legacy_models)
        return int(bool(legacy_flag) and getattr(args, 'model_name', '') in legacy_models)

    def _load_checkpoint(self, model_path, device):
        if torch.cuda.is_available():
            return torch.load(model_path)
        return torch.load(model_path, map_location=torch.device('cpu'))

    def _optimizer_checksum(self, optimizer):
        if optimizer is None:
            return 0.0, 0.0, 0.0
        exp_avg_sum = 0.0
        exp_avg_sq_sum = 0.0
        step_sum = 0.0
        for state in optimizer.state.values():
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            step = state.get('step')
            if torch.is_tensor(exp_avg):
                exp_avg_sum += float(exp_avg.detach().abs().sum().cpu())
            if torch.is_tensor(exp_avg_sq):
                exp_avg_sq_sum += float(exp_avg_sq.detach().abs().sum().cpu())
            if torch.is_tensor(step):
                step_sum += float(step.detach().sum().cpu())
            elif step is not None:
                step_sum += float(step)
        return exp_avg_sum, exp_avg_sq_sum, step_sum

    def _param_checksum(self, model):
        param_sum = 0.0
        param_sq_sum = 0.0
        with torch.no_grad():
            for param in model.parameters():
                param_sum += float(param.detach().sum().cpu())
                param_sq_sum += float((param.detach() * param.detach()).sum().cpu())
        return param_sum, param_sq_sum

    def _debug_state_checksum(self, model, snap_idx, step_flag, label):
        if not self.pisa_debug_parity:
            return
        param_sum, param_sq_sum = self._param_checksum(model)
        opt_avg_sum, opt_avg_sq_sum, opt_step_sum = self._optimizer_checksum(model.optimizer)
        self._debug_parity(
            f'[PISAParity] state label={label} snap_idx={snap_idx} step_flag={step_flag} '
            f'param_sum={param_sum:.8f} param_sq_sum={param_sq_sum:.8f} '
            f'opt_exp_avg_abs_sum={opt_avg_sum:.8f} '
            f'opt_exp_avg_sq_abs_sum={opt_avg_sq_sum:.8f} opt_step_sum={opt_step_sum:.4f}'
        )

    def _torch_rng_checksum(self):
        state = torch.random.get_rng_state()
        digest = hashlib.sha1(state.cpu().numpy().tobytes()).hexdigest()[:12]
        state_int = state.to(torch.int64)
        checksum = int(state_int.sum().item())
        return checksum, digest

    def _numpy_rng_checksum(self):
        state = np.random.get_state()[1].astype(np.uint32)
        digest = hashlib.sha1(state.tobytes()).hexdigest()[:12]
        checksum = int(state.astype(np.uint64).sum() % np.iinfo(np.int64).max)
        return checksum, digest

    def _debug_rng_checksum(self, snap_idx, step_flag, epoch, label):
        if not self.pisa_debug_parity:
            return
        torch_sum, torch_digest = self._torch_rng_checksum()
        numpy_sum, numpy_digest = self._numpy_rng_checksum()
        self._debug_parity(
            f'[PISAParity] rng label={label} snap_idx={snap_idx} step_flag={step_flag} '
            f'epoch={epoch} torch_sum={torch_sum} torch_hash={torch_digest} '
            f'numpy_sum={numpy_sum} numpy_hash={numpy_digest}'
        )

    def _debug_centroid_checksum(self, model, snap_idx, step_flag, epoch, label):
        if not self.pisa_debug_parity or not hasattr(model, 'centr') or not hasattr(model, 'centr_prev'):
            return
        centr = model.centr.detach()
        centr_prev = model.centr_prev.detach()
        centr_sum = float(centr.sum().cpu())
        centr_sq_sum = float((centr * centr).sum().cpu())
        centr_prev_sum = float(centr_prev.sum().cpu())
        centr_prev_sq_sum = float((centr_prev * centr_prev).sum().cpu())
        self._debug_parity(
            f'[PISAParity] kmeans label={label} snap_idx={snap_idx} step_flag={step_flag} '
            f'epoch={epoch} centr_shape={tuple(centr.shape)} '
            f'centr_sum={centr_sum:.8f} centr_sq_sum={centr_sq_sum:.8f} '
            f'centr_prev_sum={centr_prev_sum:.8f} centr_prev_sq_sum={centr_prev_sq_sum:.8f}'
        )

    def _pisa_epoch_seed(self, snap_idx, step_flag, epoch):
        return int(self.random_seed) + int(snap_idx) * 1000003 + int(step_flag) * 10007 + int(epoch)

    def _update_kmeans(self, model, prev_model, snap_idx):
        mode = self.pisa_kmeans_seed_mode
        if mode == 'global':
            model.update_kmeans(prev_model)
            return

        seed = self._pisa_epoch_seed(snap_idx, model.forward_flag, model.epoch)
        self._debug_parity(
            f'[PISAParity] kmeans_seed mode={mode} snap_idx={snap_idx} '
            f'step_flag={model.forward_flag} epoch={model.epoch} seed={seed}'
        )
        if mode == 'epoch':
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            model.update_kmeans(prev_model)
            return

        devices = []
        if torch.cuda.is_available() and getattr(model, '_device', None) is not None and model._device.type == 'cuda':
            devices = [model._device.index if model._device.index is not None else 0]
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            model.update_kmeans(prev_model)

    def _configure_aux_optimizer(self, model, args, snap_idx, step_flag):
        if step_flag <= 0:
            return
        mode = getattr(args, 'pisa_aux_optimizer_mode', 'reuse')
        if mode == 'reuse':
            return
        if mode == 'reset':
            model.optimizer = self._build_optimizer(model)
            self._debug_parity(
                f'[PISAParity] aux optimizer reset snap_idx={snap_idx} step_flag={step_flag}'
            )
            return
        if mode == 'load_forward':
            forward_path = f'{model.model_path}_forward_snap{snap_idx}'
            check_point = self._load_checkpoint(forward_path, model._device)
            if 'optimizer_state_dict' not in check_point:
                raise KeyError(f'No optimizer_state_dict found in {forward_path}')
            model.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            self._debug_parity(
                f'[PISAParity] aux optimizer loaded from forward checkpoint {forward_path}'
            )

    def _build_optimizer(self, model):
        if self.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        raise ValueError(f"Unknown Optimizer: {self.optimizer_name}")
    
    def write_results_excl_cold(self, model, args, snap_idx, test_loads, val_loads, hist_loads, option=''):
        """Write validation and test results to files and save metrics in JSON format."""
        compare_flag = getattr(args, 'compare_vectorized_eval', None)
        vectorized_flag = getattr(args, 'vectorized_eval', None)
        eval_msg = (
            f'Eval flags: compare_vectorized_eval={compare_flag}, '
            f'vectorized_eval={vectorized_flag}, snap_idx={snap_idx}, option={option}'
        )
        print(eval_msg, flush=True)
        logging.info(eval_msg)
        val_results = Inference.Test_excl_cold_selected(args, model, val_loads, hist_loads, label='val')
        test_results = Inference.Test_excl_cold_selected(args, model, test_loads, hist_loads, label='test')
        logging.info("Trained model testing")

        # Save validation results
        val_str = Inference.print_results(None, val_results, None)
        val_path = os.path.join(self.test_result_file, f'{option}val_snap{snap_idx}.txt')
        open(val_path, 'w+').write(val_str)

        # Save test results
        test_str = Inference.print_results(None, None, test_results)
        test_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.txt')
        open(test_path, 'w+').write(test_str)

        # Save metrics to JSON
        Ks = [10, 20, 50, 100]
        metrics = ['Recall', 'NDCG', 'MRR', 'Precision']
        json_results = {f'{metric}@{k}': v for metric, values in zip(metrics, test_results) for k, v in zip(Ks, values)}
        json_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)

    def write_results(self, model, args, corpus, snap_idx, option=''):
        """Write validation and test results to files and save metrics in JSON format."""
        v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
        t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
        logging.info("Trained model testing")

        # Save validation results
        val_str = Inference.print_results(None, v_results, None)
        val_path = os.path.join(self.test_result_file, f'{option}val_snap{snap_idx}.txt')
        open(val_path, 'w+').write(val_str)

        # Save test results
        test_str = Inference.print_results(None, None, t_results)
        test_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.txt')
        open(test_path, 'w+').write(test_str)

        # Save metrics to JSON
        Ks = [10, 20, 50, 100]
        metrics = ['Recall', 'NDCG', 'MRR', 'Precision']
        json_results = {f'{metric}@{k}': v for metric, values in zip(metrics, t_results) for k, v in zip(Ks, values)}
        json_path = os.path.join(self.test_result_file, f'{option}test_snap{snap_idx}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)

    def train(self, model, data_dict, args, corpus, prev_data, snap_idx, force_train=False, step_flag=0):
        """Main training loop with early stopping and model checkpointing."""
        logging.info(f'Training time stage: {snap_idx}')

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        
        test_loads = utils.load_data_as_dict(corpus, 'test', snap_idx)
        val_loads  = utils.load_data_as_dict(corpus, 'val', snap_idx)
        hist_loads = utils.load_data_as_dict(corpus, 'hist', snap_idx)

        # Check if model exists and handle accordingly
        model_path = f'{model.model_path}_snap{snap_idx}'
        if os.path.exists(model_path) and not force_train:
            if not (step_flag == 0 and 'plasticity' in self.dyn_method):
                model.load_model(model_path)
                #self.write_results(model, args, corpus, snap_idx)
                self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
                print(f'model already exists, skip training')
            return 0, 0
        else:
            print(f'model does not exist, training from scratch for time stage {snap_idx}')

        # Handle pretraining case
        # if snap_idx > 0 and 'pretrain' in args.dyn_method:
        #     model.load_model(f'{model.model_path}_snap0')
        #     self.write_results(model, args, corpus, snap_idx)
        #     return 0, 0

        # Load previous model for fine-tuning
        prev_model = None
        forward_model = None
        """
        step_flag: 0 -> for pure finetune -> model path suffix _forward_snap{snap_idx}
                   1 -> for PISA training -> model path suffix _snap{snap_idx}
        """
        if snap_idx == 0 and step_flag == 1:
            model.load_model(f'{model.model_path}_forward_snap0')
            model.save_model(add_path='_snap0')
            #self.write_results(model, args, corpus, snap_idx, option='')
            self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads, option='')
            return 0
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            model.load_model(f'{model.model_path}_snap{snap_idx-1}')
            model.freeze_flag = 0
            prev_model = copy.deepcopy(model)
            prev_model.eval()

            if step_flag > 0:
                forward_model = copy.deepcopy(model)
                forward_model.load_model(f'{model.model_path}_forward_snap{snap_idx}')
                forward_model.eval()

        self._configure_aux_optimizer(model, args, snap_idx, step_flag)
        self._debug_state_checksum(model, snap_idx, step_flag, 'before_training')

        # Training loop
        num_epoch = self.tepoch if ('finetune' in self.dyn_method or 'newtrain' in self.dyn_method) else self.epoch
        if snap_idx == 0:
            num_epoch = self.epoch

        best_recall = 0
        best_epoch = 0
        patience = 20
        cnt = 0
        model.forward_flag = step_flag
        self._debug_parity(
            f'[PISAParity] start snap_idx={snap_idx} step_flag={step_flag} '
            f'fast_sampler={getattr(args, "fast_sampler", 1)} '
            f'legacy_aux_neg_sampler={getattr(args, "legacy_aux_neg_sampler", 0)} '
            f'legacy_neg_sampler_effective={self._effective_legacy_neg_sampler(args)} '
            f'legacy_pisa_aux_loss={getattr(args, "legacy_pisa_aux_loss", 0)} '
            f'vectorized_eval={getattr(args, "vectorized_eval", 0)} '
            f'pisa_kmeans_seed_mode={self.pisa_kmeans_seed_mode} '
            f'pisa_aux_optimizer_mode={getattr(args, "pisa_aux_optimizer_mode", "reuse")} '
            f'shuffle={int(self.shuffle)}'
        )

        for epoch in tqdm(range(num_epoch), ncols=100, mininterval=1):
            model.epoch = epoch
            losses = self.fit(model, data_dict, prev_data, snap_idx, self.shuffle, prev_model, forward_model)
            logging.info(f'Epoch {epoch} total_loss={losses[0]:.4f} bpr_loss={losses[1]:.4f} cl_loss={losses[2]:.4f} plast_loss={losses[3]:.4f} stab_loss={losses[4]:.4f} plast_neigh_loss={losses[5]:.4f} stab_neigh_loss={losses[6]:.4f}')
            self._debug_state_checksum(model, snap_idx, step_flag, f'after_epoch_{epoch}')

            if np.isnan(losses[0]).any():
                logging.info('NaN loss, stop training')
                exit()

            # Validation and early stopping
            eval_step = args.eval_step # pisa default value 2
            patience_start_step = 0 # pisa default value 20
            if epoch >= 0 and (epoch + 1) % eval_step == 0:
                #v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
                v_results = Inference.Test_excl_cold_selected(args, model, val_loads, hist_loads, lite=True, label='val-lite')
                # gc.collect()
                current_recall = v_results[0][1]
                improved = current_recall > best_recall
                if improved:
                    best_epoch = epoch + 1
                    best_recall = current_recall
                    save_path = f'_forward_snap{snap_idx}' if (step_flag == 0 and 'plasticity' in self.dyn_method) else f'_snap{snap_idx}'
                    model.save_model(add_path=save_path)
                    cnt = 0
                else:
                    if epoch + 1 > patience_start_step:
                        cnt += eval_step
                        if cnt >= patience:
                            break
                self._debug_parity(
                    f'[PISAParity] validation snap_idx={snap_idx} step_flag={step_flag} '
                    f'epoch={epoch} recall@20={current_recall:.8f} '
                    f'best_recall@20={best_recall:.8f} best_epoch={best_epoch} '
                    f'improved={int(improved)}'
                )

        logging.info(f"Training complete. Best validation epoch: {best_epoch:03d}")
        self._debug_parity(
            f'[PISAParity] complete snap_idx={snap_idx} step_flag={step_flag} '
            f'best_epoch={best_epoch} best_recall@20={best_recall:.8f}'
        )
        
        # Load best model and write results
        model_path = f'{model.model_path}_forward_snap{snap_idx}' if (step_flag == 0 and 'plasticity' in self.dyn_method) else f'{model.model_path}_snap{snap_idx}'
        model.load_model(model_path)
        #self.write_results(model, args, corpus, snap_idx, option='forward' if step_flag == 0 and 'plasticity' in self.dyn_method else '')
        self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads, option='forward' if step_flag == 0 and 'plasticity' in self.dyn_method else '')

        return best_epoch

    def fit(self, model, data, prev_data, snap_idx, shuffle, prev_model, forward_model):
        """Single epoch training loop with k-means updates and batch processing."""
        with torch.no_grad():
            if not ('plasticity' in self.dyn_method and model.forward_flag == 0):
                self._debug_rng_checksum(snap_idx, model.forward_flag, model.epoch, 'before_kmeans')
                self._update_kmeans(model, prev_model, snap_idx)
                self._debug_rng_checksum(snap_idx, model.forward_flag, model.epoch, 'after_kmeans')
                self._debug_centroid_checksum(model, snap_idx, model.forward_flag, model.epoch, 'after_kmeans')

        # gc.collect()
        # torch.cuda.empty_cache()
        use_fast_loader = bool(getattr(data.args, 'fast_sampler', 1))
        if use_fast_loader:
            dl = utils.build_data_loader(
                data,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            )
            loader_mode = 'fast'
        else:
            if hasattr(data, '_use_fast_collate'):
                data._use_fast_collate = False
            dl = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle,
                            num_workers=self.num_workers, pin_memory=self.pin_memory)
            loader_mode = 'baseline'

        self._debug_parity(
            f'[PISAParity] loader snap_idx={snap_idx} step_flag={model.forward_flag} '
            f'epoch={model.epoch} loader={loader_mode} '
            f'fast_collate={int(getattr(data, "_use_fast_collate", False))} '
            f'legacy_aux_neg_sampler={getattr(data.args, "legacy_aux_neg_sampler", 0)} '
            f'legacy_neg_sampler_effective={int(data._use_legacy_aux_neg_sampling()) if hasattr(data, "_use_legacy_aux_neg_sampling") else 0} '
            f'num_workers={self.num_workers} persistent_workers={self.persistent_workers} '
            f'shuffle={int(bool(shuffle))}'
        )

        total_losses = []
        loss_names = ['total', 'bpr', 'cl', 'plast', 'stab', 'plast_neigh', 'stab_neigh']
        for batch_idx, current in enumerate(dl):
            #current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current = utils.batch_to_gpu(current, model._device)
            current['batch_size'] = len(current['user_id'])
            losses = self.train_recommender_vanilla(data, model, current, prev_data, snap_idx, prev_model, forward_model)
            total_losses.append(losses)
            if self.pisa_debug_parity and batch_idx < 2:
                user_sum = int(current['user_id'].detach().sum().cpu())
                item_sum = int(current['item_id'].detach().sum().cpu())
                first_user = int(current['user_id'][0].detach().cpu().reshape(-1)[0])
                first_items = current['item_id'][0].detach().cpu().reshape(-1).tolist()
                loss_msg = ' '.join(
                    f'{name}={float(np.asarray(value)):.8f}'
                    for name, value in zip(loss_names, losses)
                )
                self._debug_parity(
                    f'[PISAParity] batch snap_idx={snap_idx} step_flag={model.forward_flag} '
                    f'epoch={model.epoch} batch={batch_idx} user_sum={user_sum} '
                    f'item_sum={item_sum} first_user={first_user} '
                    f'first_items={first_items} {loss_msg}'
                )

        return [np.mean(loss).item() for loss in zip(*total_losses)]

    def train_recommender_vanilla(self, data, model, current, prev_data, time_idx, prev_model, forward_model):
        """Process a single batch of data and update model parameters."""
        model.train()
        losses = model.loss(data, current, prev_data, time_idx, prev_model, forward_model, reduction='mean')
        
        model.optimizer.zero_grad()
        losses[0].backward()
        model.optimizer.step()

        return [loss.cpu().data.numpy() for loss in losses]
