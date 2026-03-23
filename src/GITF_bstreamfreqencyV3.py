from collections import defaultdict
import torch

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
                    self, doc_count, alpha, steplowerbound, stepupperbound#, resetflag
                ):
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
            #self.resetflagAcrEpoch = resetflag
    
    def proc_newStreamDocs(self, mktbin_doc):
        if isinstance(mktbin_doc, set):
            docidset = mktbin_doc
        else:
            docidset = set()
            for d in mktbin_doc:
                docidset.add(d)
        self.global_t += 1
        self.add_streamIDsDeduped(torch.tensor(list(docidset), device=self.device), self.global_t)
    
    # ids: tensor of deduped id list
    def add_streamIDsDeduped(self, ids, t): 
        # count 1st in
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + torch.where(self.step_latest[ids] == 0, 1, self.alpha) * (t - self.step_latest[ids])
        # ignore 1st
        self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + torch.where(self.step_latest[ids] == 0, 0, torch.where(self.step_gap[ids] == 0, 1, self.alpha) ) * (t - self.step_latest[ids])
        # self.step_gap[ids] = (1 - self.alpha) * self.step_gap[ids] + self.alpha * (t - self.step_latest[ids])
        self.step_latest[ids] = t
        #self.step_latest_flag[ids] = True
    
    # def get_probability(self, idxes):
    #    return 1.0 / torch.where(self.step_gap[idxes] == 0, 1, self.step_gap[idxes])
    
    def get_minus_log_probability(self, idxes):
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
    
    # def reset(self):
    #     self.step_gap.zero_()
    #     self.step_latest.zero_()