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
                    proc_stream_mode='item_list',
                ):
        if proc_stream_mode not in {'item_set', 'item_list'}:
            raise ValueError(
                f"Unknown proc_stream_mode: {proc_stream_mode}. "
                "Expected 'item_set' or 'item_list'."
            )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            self.step_gap = torch.zeros(doc_count, device=self.device)
            #self.step_gap = torch.full((doc_count,), 2.71828, device='cuda')
            self.step_latest = torch.zeros(doc_count, device=self.device)
            #self.step_latest_flag = torch.zeros(doc_count, device='cuda', dtype=torch.bool)
            self.alpha = alpha
            self.steplowerbound = steplowerbound
            self.stepupperbound = stepupperbound
            self.const_e = torch.exp(torch.tensor(1.0, device=self.device)) # 2.71828...
            self.global_t = 0
            self.proc_stream_mode = proc_stream_mode
            #self.resetflagAcrEpoch = resetflag
    
    def proc_newStream(self, data_dict):
        if self.proc_stream_mode == 'item_set':
            self.proc_newStream_byItemset(data_dict.item_set)
        else:
            self.proc_newStream_byItemlist(data_dict.trainItem)

    def proc_newStream_byItemset(self, item_set):
        self.global_t += 1
        ids = torch.as_tensor(list(item_set), dtype=torch.long, device=self.device)
        self._update_step_state(ids, self.global_t)

    def proc_newStream_byItemlist(self, item_list):
        if len(item_list) == 0:
            raise ValueError('Cannot process an empty stream item list')
        ids_np, counts_np = np.unique(item_list, return_counts=True)
        probs_np = counts_np.astype(np.float32) / len(item_list)

        ids = torch.from_numpy(ids_np).long().to(self.device)
        probs = torch.from_numpy(probs_np).float().to(self.device)

        self.global_t += 1
        self._update_step_state_with_probs(ids, probs, self.global_t)
    
    # ids: tensor of deduped id list
    def _update_step_state(self, ids, t):
        # count 1st in
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + torch.where(self.step_latest[ids] == 0, 1, self.alpha) * (t - self.step_latest[ids])
        # ignore 1st
        self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + torch.where(self.step_latest[ids] == 0, 0, torch.where(self.step_gap[ids] == 0, 1, self.alpha) ) * (t - self.step_latest[ids])
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + self.alpha * (t - self.step_latest[ids])
        self.step_latest[ids] = t
        #self.step_latest_flag[ids] = True

    def _update_step_state_with_probs(self, ids, probs, t):
        self.step_gap[ids] = (
            (1 - self.alpha) ** (t - self.step_latest[ids]) * self.step_gap[ids]
            + torch.where(self.step_gap[ids] == 0, 1, self.alpha) * probs
        )
        self.step_latest[ids] = t
    
    # def get_probability(self, idxes):
    #    return 1.0 / torch.where(self.step_gap[idxes] == 0, 1, self.step_gap[idxes])
    
    def get_shannonInfoConent(self, idxes):
        if self.proc_stream_mode == 'item_list':
            if (self.step_gap[idxes] == 0).any():
                raise ValueError("'step_gap' has value 0")
            return self.getInfo(self.step_gap[idxes])

        # handle step_gap 0 when item 1st occurence: assign the sample initilized training weight 1 
        iw = torch.where(self.step_gap[idxes] == 0, self.const_e, self.step_gap[idxes])
        # bunded step value
        if self.steplowerbound > 1 and self.stepupperbound > 1:
            x =  torch.where(iw <= self.steplowerbound, self.steplowerbound, \
                             torch.where(iw >= self.stepupperbound, self.stepupperbound, iw))
        elif self.steplowerbound > 1:
            x =  torch.where(iw <= self.steplowerbound, self.steplowerbound, iw)
        elif self.stepupperbound > 1:
            x =  torch.where(iw >= self.stepupperbound, self.stepupperbound, iw)
        else:
            x = iw
        return  torch.log(x)

    def getInfo(self, probabilities):
        return (-torch.log2(probabilities)).clamp(0, 20.0)
    
    # def reset(self):
    #     self.step_gap.zero_()
    #     self.step_latest.zero_()
