# -*- coding: UTF-8 -*-

import os
import gc
import copy
import torch
import logging
import numpy as np
import random
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Dataloader
from typing import Dict, List, NoReturn

from utils import utils
from models.Model import Model

import matplotlib.pyplot as plt
import Inference
import json




class Runner_cons(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=200,
                            help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--l2', type=float, default=1e-04,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='',
                            help='')

        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.persistent_workers = args.persistent_workers
        self.prefetch_factor = args.prefetch_factor
        self.shuffle = bool(args.shuffle)
        self.max_grad_norm = args.max_grad_norm
        self.checkpoint_retention = args.checkpoint_retention
        pretrain_snap0 = os.path.abspath(args.pretrain_model_path + '_snap0')
        self.protected_checkpoint_paths = (
            () if 'pretrain' in args.dyn_method else (pretrain_snap0,)
        )
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None  # will store [start_time, last_step_time]

        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch


    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'adam':
            #logging.info("Optimizer: Adam")
            #if 'parameters' in self.DRM:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
            # else:
            #     optimizer = torch.optim.Adam(
            #     model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def _cleanup_previous_checkpoint(self, model, snap_idx):
        if self.checkpoint_retention == 'all' or snap_idx <= 0:
            return
        utils.remove_checkpoint(
            model.model_path + '_snap{}'.format(snap_idx - 1),
            protected_paths=self.protected_checkpoint_paths,
        )

    def make_plot(self, args, data, name, snap_idx=0):
        y = data
        x = range(len(y))
        plt.plot(x, y)
        plt.xlabel('epoch')
        plt.ylabel('{}'.format(name))
        plt.title('{}_{}'.format(name, snap_idx))
        plt.savefig(args.test_result_file+'_{}_{}.png'.format(name, snap_idx))
        plt.close()
        
    def write_results_excl_cold(self, model, args, snap_idx, test_loads, val_loads, hist_loads, option=''):
        """Write validation and test results to files and save metrics in JSON format."""
        vectorized_flag = args.vectorized_eval
        eval_msg = (
            f'Eval flags: vectorized_eval={vectorized_flag}, '
            f'snap_idx={snap_idx}, option={option}'
        )
        print(eval_msg, flush=True)
        logging.info(eval_msg)
        val_results = Inference.Test_excl_cold_selected(args, model, val_loads, hist_loads)
        test_results = Inference.Test_excl_cold_selected(args, model, test_loads, hist_loads)
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

    def write_results(self, model, args, corpus, snap_idx):
        
        v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
        t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
        logging.info("Trained model testing")

        val_str = Inference.print_results(None, v_results, None)
        val_result_filename_ = os.path.join(self.test_result_file, 'val_snap{}.txt'.format(snap_idx))
        open(val_result_filename_, 'w+').write(val_str)
        test_str = Inference.print_results(None, None, t_results)
        result_filename_ = os.path.join(self.test_result_file, 'test_snap{}.txt'.format(snap_idx))
        open(result_filename_, 'w+').write(test_str)

        ### save the test results to a json file for later analysis ###
        Ks = [10, 20, 50, 100]
        json_t = {f'{metric}@{k}': v for metric, v in zip(['Recall', 'NDCG', 'MRR', 'Precision'], t_results) for k, v in zip(Ks, v)}
        with open (os.path.join(self.test_result_file, 'test_snap{}.json'.format(snap_idx)), 'w') as f:
            json.dump(json_t, f, indent=4)

    def train(self,
              model,
              data_dict,
              args,
              corpus,
              prev_data,
              snap_idx, 
              force_train):
        
        logging.info('Training time stage: {}'.format(snap_idx))

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        
        test_loads = utils.load_data_as_dict(corpus, 'test', snap_idx)
        val_loads  = utils.load_data_as_dict(corpus, 'val', snap_idx)
        hist_loads = utils.load_data_as_dict(corpus, 'hist', snap_idx)

        # only for pretrained model (time=0) if exists
        if snap_idx == 0 and os.path.exists(model.model_path+'_snap{}'.format(0)) and force_train == False:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            # test the model
            model.load_model(model.model_path+'_snap{}'.format(snap_idx))
            #self.write_results(model, args, corpus, snap_idx)
            self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
            return 0, 0
        
        else:
            # pretrain model with time=0
            if snap_idx > 0 and 'pretrain' in args.dyn_method:
                logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
                model.load_model(model.model_path+'_snap0')
                #self.write_results(model, args, corpus, snap_idx)
                self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
                return 0, 0
            else:
                logging.info('Time_idx {} model does not exist. Start training.'.format(snap_idx))


        # Check if model exists and handle accordingly
        # model_path = f'{model.model_path}_snap{snap_idx}'
        # if os.path.exists(model_path) and not force_train:
        #     model.load_model(model_path)
        #     self.write_results(model, args, corpus, snap_idx)
        #     print(f'model already exists, skip training')
        # else:
        #     print(f'model does not exist, training from scratch for time stage {snap_idx}')


        # load previous model
        prev_model = None
        hist_model = None
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            model.load_model(model.model_path + '_snap{}'.format(snap_idx - 1))
            model.freeze_flag = 0
            prev_model = copy.deepcopy(model)
            prev_model.eval()

            if 'reinit' in args.dyn_method:
                model.add_drop.apply(model.init_weights)
                if 'stdev' in self.dyn_method:
                    torch.nn.init.normal_(model.W_prev, mean=0, std=1)
                    torch.nn.init.normal_(model.W, mean=0, std=1)
                    #torch.nn.init.normal_(model.W_hist, mean=0, std=1)
                else:
                    torch.nn.init.normal_(model.W_prev, mean=0, std=0.01)
                    torch.nn.init.normal_(model.W, mean=0, std=0.01)
                    #torch.nn.init.normal_(model.W_hist, mean=0, std=0.01)


        elif snap_idx > 0 and 'newtrain' in args.dyn_method:
            prev_model = copy.deepcopy(model)
            prev_model.load_model(model.model_path + '_snap{}'.format(snap_idx - 1))
            prev_model.eval()


        # # for lower bound minimization (plasticity)
        # club = CLUBSample(model.emb_size, model.emb_size, 256, device=model._device)
        # club.optimizer = torch.optim.Adam(club.parameters(), lr=self.learning_rate, weight_decay=self.l2)

        # Training loop
        num_epoch = self.tepoch if ('finetune' in self.dyn_method or 'newtrain' in self.dyn_method) else self.epoch
        if snap_idx == 0:
            num_epoch = self.epoch

        validation_interval_epochs = args.validation_interval_epochs
        early_stop_patience = args.early_stop_patience
        early_stop_min_delta = args.early_stop_min_delta
        best_recall = -np.inf
        best_epoch = 0
        raw_loss_history = []
        loss_names = [
            'total_loss', 'bpr_loss', 'kd_loss', 'kd_user', 'kd_item',
            'kd_user_neigh', 'kd_item_neigh',
        ]
        logging.info(
            'Early stopping: validation_interval_epochs=%d, patience=%d epochs '
            '(0 disables), min_delta=%.1e.',
            validation_interval_epochs,
            early_stop_patience,
            early_stop_min_delta,
        )
        logging.info(
            'Loss reporting: raw=sample-weighted epoch mean, smooth=%d-epoch moving average.',
            utils.LOSS_SMOOTHING_WINDOW,
        )
        logging.info(
            'Gradient safety: max_grad_norm=%g (0 disables checking and clipping).',
            self.max_grad_norm,
        )

        for epoch in tqdm(range(num_epoch), ncols=100, mininterval=1):
            model.epoch = epoch
            raw_losses, losses_are_finite = self.fit(
                model, data_dict, prev_data, snap_idx, self.shuffle, prev_model
            )
            if not losses_are_finite:
                logging.info(
                    'Epoch %d encountered a non-finite loss or gradient; stop training.',
                    epoch,
                )
                if best_epoch == 0:
                    raise FloatingPointError(
                        'Non-finite loss or gradient before the first validation '
                        f'checkpoint at snapshot {snap_idx}'
                    )
                break

            raw_losses = np.asarray(raw_losses, dtype=np.float64)
            raw_loss_history.append(raw_losses)
            smooth_losses = np.mean(
                np.stack(raw_loss_history[-utils.LOSS_SMOOTHING_WINDOW:]), axis=0
            )
            loss_msg = ' '.join(
                f'raw_{name}={raw:.4f} smooth_{name}={smooth:.4f}'
                for name, raw, smooth in zip(loss_names, raw_losses, smooth_losses)
            )
            logging.info(f'Epoch {epoch} {loss_msg}')

            # Validation and early stopping
            completed_epochs = epoch + 1
            should_validate = (
                completed_epochs % validation_interval_epochs == 0
                or completed_epochs == num_epoch
            )
            if not should_validate:
                continue

            v_results = Inference.Test_excl_cold_selected(
                args, model, val_loads, hist_loads, lite=True
            )
            current_recall = v_results[0][1]
            if current_recall > best_recall + early_stop_min_delta:
                best_epoch = completed_epochs
                best_recall = current_recall
                model.save_model(add_path=f'_snap{snap_idx}')
            else:
                epochs_without_improvement = completed_epochs - best_epoch
                if (
                    early_stop_patience > 0
                    and epochs_without_improvement >= early_stop_patience
                ):
                    logging.info(
                        'Early stopping at epoch %d: Recall@20 did not improve by more than '
                        '%.1e for %d epochs (best=%.8f at epoch %d).',
                        completed_epochs,
                        early_stop_min_delta,
                        epochs_without_improvement,
                        best_recall,
                        best_epoch,
                    )
                    break

        logging.info(f"Training complete. Best validation epoch: {best_epoch:03d}")
        
        # Load best model and write results
        model_path = f'{model.model_path}_snap{snap_idx}'
        model.load_model(model_path)
        #self.write_results(model, args, corpus, snap_idx)
        self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
        self._cleanup_previous_checkpoint(model, snap_idx)

        return best_epoch



    def fit(self, model, data, prev_data, snap_idx, shuffle, prev_model):

        if 'piw' in self.dyn_method and snap_idx > 0:
            with torch.no_grad():
                model.update_kmeans(prev_model)


        # gc.collect()
        # torch.cuda.empty_cache()
        dl = utils.build_data_loader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        weighted_loss_sums = None
        sample_count = 0
        for current in dl:
            #current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current = utils.batch_to_gpu(current, model._device)
            num_samples = len(current['user_id'])
            losses, losses_are_finite = self.train_recommender_vanilla(
                data, model, current, prev_data, snap_idx, prev_model
            )
            if not losses_are_finite:
                return None, False

            batch_losses = np.asarray(losses, dtype=np.float64)
            if weighted_loss_sums is None:
                weighted_loss_sums = np.zeros_like(batch_losses)
            weighted_loss_sums += batch_losses * num_samples
            sample_count += num_samples

        if sample_count == 0:
            raise RuntimeError('Training DataLoader produced no samples')
        return (weighted_loss_sums / sample_count).tolist(), True


    def train_recommender_vanilla(self, data, model, current, prev_data, time_idx, prev_model):
        """Process a single batch of data and update model parameters."""
        model.train()
        losses = model.loss(data, current, prev_data, time_idx, prev_model, reduction='mean')
        loss_values = [loss.detach().cpu().item() for loss in losses]
        if not all(torch.isfinite(loss).all() for loss in losses):
            return loss_values, False
        
        model.optimizer.zero_grad(set_to_none=True)
        losses[0].backward()
        if self.max_grad_norm > 0:
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.max_grad_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                logging.warning(
                    'Gradient norm check failed at snapshot %d: %s',
                    time_idx,
                    error,
                )
                model.optimizer.zero_grad(set_to_none=True)
                return loss_values, False
        model.optimizer.step()

        return loss_values, True
