# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import Model


class LightCCF(Model):
    """LightCCF: Light Contrastive Collaborative Filtering (SIGIR'25).

    MF-encoder variant: raw user/item embeddings (no graph convolution);
    the neighbor-aggregation contrastive loss performs the aggregation.
    Total loss = BPR (with optional GI stream weights) + ssl_lambda *
    neighbor-aggregate InfoNCE + reg_lambda * L2 regularization,
    following ZzYUuuu/LightCCF.

    Faithful reproduction: main.py disables the shared Adam weight decay
    (--l2) for this model, so regularization is controlled solely by
    --lightccf_reg_lambda.
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--lightccf_tau', type=float, default=0.24,
                            help='Temperature of the neighbor-aggregate contrastive loss.')
        parser.add_argument('--lightccf_ssl_lambda', type=float, default=5.0,
                            help='Weight of the neighbor-aggregate contrastive loss.')
        parser.add_argument('--lightccf_reg_lambda', type=float, default=1e-4,
                            help='Weight of the explicit L2 regularization loss.')
        return Model.parse_model_args(parser)

    @staticmethod
    def init_weights(m):
        if 'Embedding' in str(type(m)):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def __init__(self, args, corpus, data, time_idx):
        self.emb_size = args.emb_size
        self.lightccf_tau = args.lightccf_tau
        self.lightccf_ssl_lambda = args.lightccf_ssl_lambda
        self.lightccf_reg_lambda = args.lightccf_reg_lambda
        self.time_idx = time_idx
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def computer(self):
        return self.u_embeddings.weight, self.i_embeddings.weight

    def forward(self, u_ids, i_ids):
        u_ids = u_ids.to(torch.long)
        i_ids = i_ids.to(torch.long)
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
        return prediction

    def infer_user_scores(self, u_ids, i_ids):
        u_ids = u_ids.to(torch.long)
        i_ids = i_ids.to(torch.long)
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        scores = torch.matmul(cf_u_vectors, cf_i_vectors.t())
        return scores

    def _get_neighbor_aggregate_loss(self, user_vectors, pos_vectors):
        """InfoNCE-style loss pulling each user toward its positive item
        while repelling from all other batch pairs and the users themselves."""
        u_vectors = F.normalize(user_vectors, dim=-1)
        p_vectors = F.normalize(pos_vectors, dim=-1)
        pos_score = torch.exp((u_vectors * p_vectors).sum(dim=-1) / self.lightccf_tau)
        total_score = (
            torch.matmul(u_vectors, p_vectors.transpose(0, 1))
            + torch.matmul(u_vectors, u_vectors.transpose(0, 1))
        )
        total_score = torch.exp(total_score / self.lightccf_tau).sum(dim=1)
        # 10e-6 == 1e-5, matching the original implementation
        return -torch.log(pos_score / total_score + 10e-6)

    def _get_reg_loss(self, user_vectors, pos_vectors, neg_vectors):
        """L2 regularization over the batch's raw embeddings (original formula)."""
        reg_loss = 0
        for embedding in (user_vectors, pos_vectors, neg_vectors):
            reg_loss = reg_loss + 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])
        return reg_loss

    def loss(self, current_data, reduction, gitf_w=None):
        # BPR term with GI stream weights, inherited from the base model
        bpr_loss = super().loss(current_data, reduction, gitf_w)

        all_users, all_items = self.computer()
        i_ids = current_data['item_id'].to(torch.long)
        u_ids = current_data['user_id'].reshape(-1).to(torch.long)
        pos_ids = i_ids[:, 0]
        neg_ids = i_ids[:, 1:1 + self.num_neg]

        u_vectors = all_users[u_ids]
        p_vectors = all_items[pos_ids]
        n_vectors = all_items[neg_ids.reshape(-1)]

        na_loss = self._get_neighbor_aggregate_loss(u_vectors, p_vectors)
        if reduction == 'mean':
            na_loss = na_loss.mean()
        reg_loss = self._get_reg_loss(u_vectors, p_vectors, n_vectors)

        return (
            bpr_loss
            + self.lightccf_ssl_lambda * na_loss
            + self.lightccf_reg_lambda * reg_loss
        )
