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


class Runner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=10,
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
        
        # for global info-content training framework (gitf)
        parser.add_argument("--strefreq_alpha", type=float, default=0.01)
        parser.add_argument("--strefreq_steplowerbound", type=float, default=-1) # 2.71828
        parser.add_argument("--strefreq_stepupperbound", type=float, default=-1) # 1096
        parser.add_argument(
            "--gitf_weight_mode",
            type=str,
            default="margin",
            choices=["margin", "example"],
            help="Apply GI weights to the BPR score margin or to each example loss.",
        )

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
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None  # will store [start_time, last_step_time]

        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch

        # for global info-content training framework (gitf)
        self.stream_frequency = None
        if 'globalinfocontent' in args.dyn_method:
            from GITF_bstreamfreqencyV3  import bStreamFrequencyV3
            self.stream_frequency = bStreamFrequencyV3(
                corpus.n_items, 
                args.strefreq_alpha, 
                args.strefreq_steplowerbound, 
                args.strefreq_stepupperbound,
            )


    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        if self.optimizer_name.lower() != 'adam':
            raise ValueError(f"Unknown Optimizer: {self.optimizer_name}")
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)



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
        Ks = [10,20,50,100]
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

        if self.stream_frequency:
            logging.info('doing global info-content for stage: {}'.format(snap_idx))
            self.stream_frequency.proc_newStreamDocs(data_dict.item_set)

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        test_loads = utils.load_data_as_dict(corpus, 'test', snap_idx)
        val_loads  = utils.load_data_as_dict(corpus, 'val', snap_idx)
        hist_loads = utils.load_data_as_dict(corpus, 'hist', snap_idx)
        
        if snap_idx == 0 and os.path.exists(model.model_path+'_snap{}'.format(0)) and force_train == False:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            # test the model
            model.load_model(model.model_path+'_snap{}'.format(snap_idx))
            #self.write_results(model, args, corpus, snap_idx)
            self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
            return 0, 0
        
        ### assuming existing models are properly trained ###
        elif os.path.exists(model.model_path+'_snap{}'.format(snap_idx)) and force_train == False:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            model.load_model(model.model_path+'_snap{}'.format(snap_idx))
            #self.write_results(model, args, corpus, snap_idx)
            self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)
            #self.write_results_for_different_user_group(model, args, corpus, snap_idx)

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


        # load previous model
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            #print(model.model_path+'_snap{}'.format(idx-1))
            model.load_model(model.model_path+'_snap{}'.format(snap_idx-1))

        self._check_time(start=True)
        self.time_d = {}
        logging.info('dyn_method: {}'.format(self.dyn_method))
        if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method:
            num_epoch = self.tepoch
            shuffle = self.shuffle
            if snap_idx == 0:
                num_epoch = self.epoch
                shuffle = self.shuffle
        elif 'fulltrain' in self.dyn_method or 'pretrain' in self.dyn_method:
            num_epoch = self.epoch
            shuffle = self.shuffle

        validation_interval_epochs = args.validation_interval_epochs
        early_stop_patience = args.early_stop_patience
        early_stop_min_delta = args.early_stop_min_delta
        best_recall = -np.inf
        best_epoch = 0
        raw_loss_history = []
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

        titer = tqdm(range(num_epoch), ncols=300)
        for epoch in titer:
            self._check_time()
            raw_loss, loss_is_finite = self.fit(
                model, data_dict, prev_data, snap_idx, shuffle
            )
            training_time = self._check_time()

            if not loss_is_finite:
                logging.info(
                    'Epoch %d encountered a non-finite loss or gradient; '
                    'stop training. [%.1f s]',
                    epoch + 1,
                    training_time,
                )
                if best_epoch == 0:
                    raise FloatingPointError(
                        'Non-finite loss or gradient before the first validation '
                        f'checkpoint at snapshot {snap_idx}'
                    )
                break

            raw_loss_history.append(raw_loss)
            smooth_loss = float(np.mean(
                raw_loss_history[-utils.LOSS_SMOOTHING_WINDOW:]
            ))

            logging.info(
                'Epoch {:<3} raw_loss={:<.4f} smooth_loss={:<.4f} [{:<.1f} s]'.format(
                    epoch + 1, raw_loss, smooth_loss, training_time
                )
            )

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
                model.save_model(add_path='_snap{}'.format(snap_idx))
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
        
        logging.info("End train and valid. Best validation epoch is {:03d}.".format(best_epoch))
        model.load_model(model.model_path+'_snap{}'.format(snap_idx))
        #self.write_results(model, args, corpus, snap_idx)
        self.write_results_excl_cold(model, args, snap_idx, test_loads, val_loads, hist_loads)

        return self.time[1] - self.time[0], best_epoch


    def fit(self, model, data, prev_data, snap_idx, shuffle):
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
        
        weighted_loss_sum = 0.0
        sample_count = 0
        for current in dl:
            #current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current = utils.batch_to_gpu(current, model._device)
            num_samples = len(current['user_id'])
            batch_loss_value, loss_is_finite = self.train_recommender_vanilla(
                dl, model, current, prev_data, snap_idx
            )
            if not loss_is_finite:
                return None, False

            weighted_loss_sum += batch_loss_value * num_samples
            sample_count += num_samples

        if sample_count == 0:
            raise RuntimeError('Training DataLoader produced no samples')
        return weighted_loss_sum / sample_count, True

    def train_recommender_vanilla(self, data, model, current, prev_data,time_idx):
        # Train recommender
        model.train()
        # Get recommender's prediction and loss from the ``current'' data at t
        #u_ids, i_ids, prev_data, data, snap_idx,reduction='mean'
        #self, data, prev_data, snap_idx,reduction
        if self.stream_frequency:
            pos_items = current['item_id'][:,:1].squeeze(-1)
            w = self.stream_frequency.get_minus_log_probability(pos_items)
        else:
            w = None
        total_loss = model.loss(current, reduction='mean', gitf_w=w)
        loss_value = total_loss.detach().cpu().item()
        if not torch.isfinite(total_loss).all():
            return loss_value, False

        # Update the recommender
        model.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
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
                return loss_value, False
        model.optimizer.step()

        return loss_value, True
