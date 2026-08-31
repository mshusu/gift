import torch
import numpy as np

"""
V3: give everyone 0 initilly
    Given an item i,
    - its first occurence
        do not set to its period step_gap
        give it a weight 1
    - starting from its second occurence
        its period step_gap =  (t - step_latest) 
                               or 
                               (1 - alpha) * step_gap + alpha * (t - step_latest)
"""
class bStreamFrequencyV3:
    def __init__(
        self, doc_count, alpha, steplowerbound, stepupperbound,
                    proc_stream_mode='item_list_infoPrEMA', pro_alpha=1.0,
                ):
        valid_modes = {
            'item_set_infoFr',
            'item_set_infoPr',
            'item_list_infoPrEMA',
            'item_list_infoGlobal',
            'item_list_parEntropyPrEMA',
            'item_list_parEntropyEMA',
            'item_list_parEntropyGlobal',
            'item_both_infoFr',
            'item_both_infoPr',
            'item_both_partialEntropy',
        }
        if proc_stream_mode not in valid_modes:
            raise ValueError(
                f"Unknown proc_stream_mode: {proc_stream_mode}. "
                f'Expected one of {sorted(valid_modes)}.'
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            self.step_gap = torch.zeros(doc_count, device=self.device)
            #self.step_gap = torch.full((doc_count,), 2.71828, device='cuda')
            self.step_latest = torch.zeros(doc_count, device=self.device)
            #self.step_latest_flag = torch.zeros(doc_count, device='cuda', dtype=torch.bool)
            self.alpha = alpha
            self.pro_alpha = pro_alpha
            self.steplowerbound = steplowerbound
            self.stepupperbound = stepupperbound
            self.const_e = torch.exp(torch.tensor(1.0, device=self.device)) # 2.71828...
            self.global_t = 0
            self.total_occurrence = 0
            self.total_snapshot_blocks = 0
            self.proc_stream_mode = proc_stream_mode
            if proc_stream_mode in {
                'item_both_infoFr',
                'item_both_infoPr',
                'item_both_partialEntropy',
            }:
                self.step_gap_itemset = torch.zeros(doc_count, device=self.device)
                self.step_gap_itemlist_ema = torch.zeros(doc_count, device=self.device)
                self.step_latest_itemlist_ema = torch.zeros(doc_count, device=self.device)
            if proc_stream_mode in {
                'item_both_infoFr',
                'item_both_partialEntropy',
            }:
                self.step_latest_itemset = torch.zeros(doc_count, device=self.device)
            #self.resetflagAcrEpoch = resetflag
    
    def proc_newStream(self, data_dict):
        if self.proc_stream_mode == 'item_set_infoPr':
            return self.proc_newStream_byItemset_Pr(data_dict.item_set)

        if self.proc_stream_mode == 'item_set_infoFr':
            return self.proc_newStream_byItemset_Fr(data_dict.item_set)

        if self.proc_stream_mode == 'item_both_infoPr':
            self.proc_newStream_byItemset_Pr(
                data_dict.item_set,
                step_gap=self.step_gap_itemset,
            )
            return self.proc_newStream_byItemlist_infoEMA(
                data_dict.trainItem,
                step_gap=self.step_gap_itemlist_ema,
                step_latest=self.step_latest_itemlist_ema,
                increment_time=False,
            )

        if self.proc_stream_mode in {
            'item_both_infoFr',
            'item_both_partialEntropy',
        }:
            self.proc_newStream_byItemset_Fr(
                data_dict.item_set,
                step_gap=self.step_gap_itemset,
                step_latest=self.step_latest_itemset,
            )
            return self.proc_newStream_byItemlist_infoEMA(
                data_dict.trainItem,
                step_gap=self.step_gap_itemlist_ema,
                step_latest=self.step_latest_itemlist_ema,
                increment_time=False,
            )

        if self.proc_stream_mode in {
            'item_list_infoPrEMA',
            'item_list_parEntropyPrEMA',
        }:
            return self.proc_newStream_byItemlist_infoEMA(data_dict.trainItem)

        if self.proc_stream_mode == 'item_list_parEntropyEMA':
            return self.proc_newStream_byItemlist_parEntropyEMA(data_dict.trainItem)

        if self.proc_stream_mode in {
            'item_list_infoGlobal',
            'item_list_parEntropyGlobal',
        }:
            return self.proc_newStream_byItemlist_infoGlobal(data_dict.trainItem)

        raise ValueError(f'Unsupported proc_stream_mode: {self.proc_stream_mode}')

    def proc_newStream_byItemset_Fr(
        self,
        item_set,
        step_gap=None,
        step_latest=None,
        increment_time=True,
    ):
        if increment_time:
            self.global_t += 1
        step_gap = self.step_gap if step_gap is None else step_gap
        step_latest = self.step_latest if step_latest is None else step_latest
        ids = torch.as_tensor(list(item_set), dtype=torch.long, device=self.device)
        self._update_step_state(ids, self.global_t, step_gap, step_latest)

    def proc_newStream_byItemset_Pr(self, item_set, step_gap=None):
        self.global_t += 1
        self.total_snapshot_blocks += 1

        step_gap = self.step_gap if step_gap is None else step_gap
        ids = torch.as_tensor(list(item_set), dtype=torch.long, device=self.device)
        counts = torch.ones(
            len(ids),
            dtype=step_gap.dtype,
            device=self.device,
        )
        self._update_step_state_with_counts(ids, counts, step_gap=step_gap)

    def proc_newStream_byItemlist_infoEMA(
        self,
        item_list,
        step_gap=None,
        step_latest=None,
        increment_time=True,
    ):
        if len(item_list) == 0:
            raise ValueError('Cannot process an empty stream item list')
        ids_np, counts_np = np.unique(item_list, return_counts=True)
        probs_np = counts_np.astype(np.float32) / len(item_list)

        ids = torch.from_numpy(ids_np).long().to(self.device)
        probs = torch.from_numpy(probs_np).float().to(self.device)

        if increment_time:
            self.global_t += 1
        step_gap = self.step_gap if step_gap is None else step_gap
        step_latest = self.step_latest if step_latest is None else step_latest
        self._update_step_state_with_probs(
            ids,
            probs,
            self.global_t,
            step_gap,
            step_latest,
        )

    def proc_newStream_byItemlist_parEntropyEMA(
        self,
        item_list,
        step_gap=None,
        step_latest=None,
        increment_time=True,
    ):
        if len(item_list) == 0:
            raise ValueError('Cannot process an empty stream item list')
        ids_np, counts_np = np.unique(item_list, return_counts=True)
        probs_np = counts_np.astype(np.float32) / len(item_list)

        ids = torch.from_numpy(ids_np).long().to(self.device)
        probs = torch.from_numpy(probs_np).float().to(self.device)
        partial_entropy = probs * self.getInfo(probs)

        if increment_time:
            self.global_t += 1
        step_gap = self.step_gap if step_gap is None else step_gap
        step_latest = self.step_latest if step_latest is None else step_latest
        self._update_step_state_with_partialEntropy(
            ids,
            partial_entropy,
            self.global_t,
            step_gap,
            step_latest,
        )

    def proc_newStream_byItemlist_infoGlobal(self, item_list):
        if len(item_list) == 0:
            raise ValueError('Cannot process an empty stream item list')
        ids_np, counts_np = np.unique(item_list, return_counts=True)
        ids = torch.from_numpy(ids_np).long().to(self.device)
        counts = torch.from_numpy(counts_np).to(
            device=self.device,
            dtype=self.step_gap.dtype,
        )

        self.global_t += 1
        self.total_occurrence += len(item_list)
        self._update_step_state_with_counts(ids, counts)
    
    # ids: tensor of deduped id list
    def _update_step_state(self, ids, t, step_gap, step_latest):
        # count 1st in
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + torch.where(self.step_latest[ids] == 0, 1, self.alpha) * (t - self.step_latest[ids])
        # ignore 1st
        step_gap[ids] = (1 - self.alpha) * step_gap[ids] + torch.where(step_latest[ids] == 0, 0, torch.where(step_gap[ids] == 0, 1, self.alpha) ) * (t - step_latest[ids])
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + self.alpha * (t - self.step_latest[ids])
        step_latest[ids] = t
        #self.step_latest_flag[ids] = True

    def _update_step_state_with_probs(self, ids, probs, t, step_gap, step_latest):
        step_gap[ids] = (
            (1 - self.pro_alpha) ** (t - step_latest[ids]) * step_gap[ids]
            + torch.where(step_gap[ids] == 0, 1, self.pro_alpha) * probs
        )
        step_latest[ids] = t

    def _update_step_state_with_partialEntropy(
        self,
        ids,
        partial_entropy,
        t,
        step_gap,
        step_latest,
    ):
        step_gap[ids] = (
            (1 - self.pro_alpha) ** (t - step_latest[ids]) * step_gap[ids]
            + torch.where(step_latest[ids] == 0, 1, self.pro_alpha)
            * partial_entropy
        )
        step_latest[ids] = t

    def _update_step_state_with_counts(self, ids, counts, step_gap=None):
        step_gap = self.step_gap if step_gap is None else step_gap
        step_gap[ids] += counts
    
    def get_streamWeight(self, idxes):
        if self.proc_stream_mode == 'item_set_infoPr':
            return self._get_shannonInfo_byItemset_Pr(idxes)

        if self.proc_stream_mode == 'item_both_infoPr':
            return (
                self._get_shannonInfo_byItemset_Pr(
                    idxes,
                    step_gap=self.step_gap_itemset,
                )
                + self._get_shannonInfo_byItemlist_infoEMA(
                    idxes,
                    step_gap=self.step_gap_itemlist_ema,
                )
            )

        if self.proc_stream_mode == 'item_set_infoFr':
            return self._get_shannonInfo_byItemset(idxes)

        if self.proc_stream_mode == 'item_both_infoFr':
            return (
                self._get_shannonInfo_byItemset(
                    idxes,
                    step_gap=self.step_gap_itemset,
                )
                + self._get_shannonInfo_byItemlist_infoEMA(
                    idxes,
                    step_gap=self.step_gap_itemlist_ema,
                )
            )

        if self.proc_stream_mode == 'item_both_partialEntropy':
            return self._get_partialEntropy_byItemset_andItemlist_infoEMA(idxes)

        if self.proc_stream_mode == 'item_list_infoPrEMA':
            return self._get_shannonInfo_byItemlist_infoEMA(idxes)

        if self.proc_stream_mode == 'item_list_infoGlobal':
            return self._get_shannonInfo_byItemlist_infoGlobal(idxes)

        if self.proc_stream_mode == 'item_list_parEntropyPrEMA':
            return self._get_partialEntropy_byItemlist_EMA(idxes)

        if self.proc_stream_mode == 'item_list_parEntropyEMA':
            return self._get_partialEntropy_byItemlist_parEntropyEMA(idxes)

        if self.proc_stream_mode == 'item_list_parEntropyGlobal':
            return self._get_partialEntropy_byItemlist_Global(idxes)

        raise ValueError(f'Unsupported proc_stream_mode: {self.proc_stream_mode}')

    def _get_shannonInfo_byItemset(self, idxes, step_gap=None):
        information, _ = self._get_itemset_info_and_probability(idxes, step_gap)
        return information

    def _get_shannonInfo_byItemset_Pr(self, idxes, step_gap=None):
        if self.total_snapshot_blocks == 0:
            raise ValueError(
                'Cannot calculate information before processing a stream'
            )

        step_gap = self.step_gap if step_gap is None else step_gap
        counts = step_gap[idxes]
        if (counts <= 0).any():
            raise ValueError("'step_gap' has a non-positive value")

        total = torch.tensor(
            self.total_snapshot_blocks,
            dtype=counts.dtype,
            device=counts.device,
        )
        return torch.log2(total) - torch.log2(counts)

    def _get_shannonInfo_byItemlist_infoEMA(self, idxes, step_gap=None):
        probabilities = self._get_probabilities_byItemlist_EMA(idxes, step_gap)
        return self.getInfo(probabilities)

    def _get_shannonInfo_byItemlist_infoGlobal(self, idxes):
        counts, total = self._get_count_state_byItemlist_Global(idxes)
        return torch.log2(total) - torch.log2(counts)

    def _get_partialEntropy_byItemlist_EMA(self, idxes):
        probabilities = self._get_probabilities_byItemlist_EMA(idxes)
        return probabilities * self.getInfo(probabilities)

    def _get_partialEntropy_byItemlist_parEntropyEMA(self, idxes):
        partial_entropy = self.step_gap[idxes]
        if (partial_entropy < 0).any():
            raise ValueError("'step_gap' has a negative value")
        return partial_entropy

    def _get_partialEntropy_byItemlist_Global(self, idxes):
        counts, total = self._get_count_state_byItemlist_Global(idxes)
        probabilities = counts / total
        information = torch.log2(total) - torch.log2(counts)
        return probabilities * information

    def _get_partialEntropy_byItemset_andItemlist_infoEMA(self, idxes):
        itemset_info, itemset_probability = self._get_itemset_info_and_probability(
            idxes,
            self.step_gap_itemset,
        )
        ema_probability = self._get_probabilities_byItemlist_EMA(
            idxes,
            self.step_gap_itemlist_ema,
        )
        ema_info = self.getInfo(ema_probability)
        return (
            itemset_probability * itemset_info
            + ema_probability * ema_info
        )

    def _get_itemset_info_and_probability(self, idxes, step_gap=None):
        # Handle step_gap 0 for first occurrences by assigning weight 1.
        step_gap = self.step_gap if step_gap is None else step_gap
        iw = torch.where(step_gap[idxes] == 0, self.const_e, step_gap[idxes])
        iw = self._apply_step_bounds(iw)
        return torch.log(iw), 1.0 / iw

    def _get_probabilities_byItemlist_EMA(self, idxes, step_gap=None):
        step_gap = self.step_gap if step_gap is None else step_gap
        probabilities = step_gap[idxes]
        if (probabilities <= 0).any():
            raise ValueError("'step_gap' has a non-positive value")
        return probabilities

    def _get_count_state_byItemlist_Global(self, idxes):
        if self.total_occurrence == 0:
            raise ValueError(
                'Cannot calculate information before processing a stream'
            )

        counts = self.step_gap[idxes]
        if (counts <= 0).any():
            raise ValueError("'step_gap' has a non-positive value")

        total = torch.tensor(
            self.total_occurrence,
            dtype=counts.dtype,
            device=counts.device,
        )
        return counts, total

    def _apply_step_bounds(self, values):
        if self.steplowerbound > 1 and self.stepupperbound > 1:
            return torch.where(
                values <= self.steplowerbound,
                self.steplowerbound,
                torch.where(
                    values >= self.stepupperbound,
                    self.stepupperbound,
                    values,
                ),
            )
        elif self.steplowerbound > 1:
            return torch.where(values <= self.steplowerbound, self.steplowerbound, values)
        elif self.stepupperbound > 1:
            return torch.where(values >= self.stepupperbound, self.stepupperbound, values)
        return values

    def getInfo(self, probabilities):
        return -torch.log2(probabilities)
    
    # def reset(self):
    #     self.step_gap.zero_()
    #     self.step_latest.zero_()
