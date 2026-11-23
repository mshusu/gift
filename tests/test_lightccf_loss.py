import math
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))

# The loss tests do not need the data reader or plotting backend.
reader_module = types.ModuleType('helpers.Reader')
reader_module.Reader = object
sys.modules['helpers.Reader'] = reader_module
matplotlib_module = types.ModuleType('matplotlib')
pyplot_module = types.ModuleType('matplotlib.pyplot')
matplotlib_module.pyplot = pyplot_module
sys.modules['matplotlib'] = matplotlib_module
sys.modules['matplotlib.pyplot'] = pyplot_module

from models.Model import Model
from models.general.LightCCF import LightCCF


def resolve_na_layer(na_layer, weight_mode):
    # mirrors LightCCF.__init__ 'auto' resolution
    if na_layer == 'auto':
        return 'loss' if weight_mode == 'example' else 'prob'
    return na_layer


def build_lightccf(weight_mode='margin', ssl_lambda=0.0, reg_lambda=0.0, tau=1.0, num_neg=2,
                   na_layer='auto'):
    model = object.__new__(LightCCF)
    model.num_neg = num_neg
    model.gitf_weight_mode = weight_mode
    model.gitf_na_layer = resolve_na_layer(na_layer, weight_mode)
    model.lightccf_tau = tau
    model.lightccf_ssl_lambda = ssl_lambda
    model.lightccf_reg_lambda = reg_lambda
    return model


def build_base_model(weight_mode='margin', num_neg=2):
    model = object.__new__(Model)
    model.num_neg = num_neg
    model.gitf_weight_mode = weight_mode
    return model


# users 0, 1 ; items 0..5, dot products:
# user 0 (1,0): item 0 -> 1.0, item 1 -> 0.5, item 2 -> 0.0
# user 1 (0,1): item 3 -> 2.0, item 4 -> 1.0, item 5 -> 1.0
ALL_USERS = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
ALL_ITEMS = torch.tensor([
    [1.0, 0.0],
    [0.5, 0.0],
    [0.0, 0.0],
    [0.0, 2.0],
    [0.0, 1.0],
    [0.0, 1.0],
])


def stub_computer(model):
    model.computer = lambda: (ALL_USERS, ALL_ITEMS)
    return model


def current_data():
    return {
        'user_id': torch.tensor([[0], [1]]),
        'item_id': torch.tensor([[0, 1, 2], [3, 4, 5]]),
    }


# batch users 0, 1 with positives 0, 3: after normalization the positives are
# (1,0) and (0,1), so for each row the diagonal terms dominate the denominator:
# total = exp(2) + exp(0) = e^2 + 1, pos score = e
NA_RATIO = math.e / (math.e ** 2 + 1)
NA_UNWEIGHTED = -math.log(NA_RATIO + 1e-5)


class NeighborAggregateLossTests(unittest.TestCase):
    def test_hand_computed_value(self):
        model = build_lightccf(tau=1.0)
        users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        na = model._get_neighbor_aggregate_loss(users, positives)
        # pos score e, denominator exp(2)+exp(0) = e^2+1 per row
        expected = -math.log(math.e / (math.e ** 2 + 1) + 1e-5)
        self.assertAlmostEqual(na.mean().item(), expected, places=3)

    def test_temperature_scales_scores(self):
        model = build_lightccf(tau=2.0)
        users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        na = model._get_neighbor_aggregate_loss(users, positives)
        # pos score exp(1/2), denominator exp(2/2)+exp(0/2) = e+1
        expected = -math.log(math.e ** 0.5 / (math.e + 1) + 1e-5)
        self.assertAlmostEqual(na.mean().item(), expected, places=3)


class LightCCFLossTests(unittest.TestCase):
    def test_matches_base_bpr_when_aux_terms_disabled(self):
        current = current_data()
        weights = torch.tensor([0.5, 2.0])
        base = stub_computer(build_base_model('margin'))
        lightccf = stub_computer(build_lightccf('margin', ssl_lambda=0.0, reg_lambda=0.0))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights)
        lightccf_loss = lightccf.loss(current, reduction='mean', gitf_w=weights)
        self.assertTrue(torch.isclose(base_loss, lightccf_loss, atol=1e-6))

    def test_margin_both_inherited(self):
        # equal weights: margin_both must reduce to margin (inherited from base)
        current = current_data()
        margin = stub_computer(build_lightccf('margin', ssl_lambda=0.0, reg_lambda=0.0))
        margin_both = stub_computer(build_lightccf('margin_both', ssl_lambda=0.0, reg_lambda=0.0))
        loss_margin = margin.loss(
            current, reduction='mean', gitf_w=torch.full((2,), 1.7)
        )
        loss_margin_both = margin_both.loss(
            current, reduction='mean', gitf_w=torch.full((2, 3), 1.7)
        )
        self.assertTrue(torch.isclose(loss_margin, loss_margin_both, atol=1e-6))

    def test_ssl_term_shifts_loss_by_hand_computed_value(self):
        current = current_data()
        base = stub_computer(build_base_model('margin'))
        base_loss = base.loss(current, reduction='mean', gitf_w=None).item()
        model = stub_computer(build_lightccf('margin', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0))
        with_ssl = model.loss(current, reduction='mean', gitf_w=None).item()
        self.assertAlmostEqual(with_ssl, base_loss + 2.0 * NA_UNWEIGHTED, places=3)

    def test_ssl_margin_mode_weights_prob(self):
        # margin mode + auto layer: the positive's softmax probability inside
        # the log is weighted (weights must not average to 1, or the weighting
        # cancels in the mean)
        current = current_data()
        weights = torch.tensor([0.5, 1.5])
        base = stub_computer(build_base_model('margin'))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights).item()
        model = stub_computer(build_lightccf('margin', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0))
        weighted = model.loss(current, reduction='mean', gitf_w=weights).item()
        expected_na = (
            -math.log(0.5 * NA_RATIO + 1e-5) - math.log(1.5 * NA_RATIO + 1e-5)
        ) / 2
        self.assertAlmostEqual(weighted, base_loss + 2.0 * expected_na, places=4)

    def test_na_layer_none_ignores_weights(self):
        users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weights = torch.tensor([0.5, 2.0])
        model = build_lightccf('margin', tau=1.0, na_layer='none')
        with_weights = model._get_neighbor_aggregate_loss(users, positives, weights)
        without_weights = model._get_neighbor_aggregate_loss(users, positives, None)
        self.assertTrue(torch.isclose(with_weights, without_weights, atol=1e-6).all())

    def test_na_layer_loss_weights_output_loss(self):
        # explicit loss layer: per-example na loss multiplied by the weight
        current = current_data()
        weights = torch.tensor([0.5, 2.0])
        base = stub_computer(build_base_model('margin'))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights).item()
        model = stub_computer(build_lightccf('margin', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0,
                                             na_layer='loss'))
        weighted = model.loss(current, reduction='mean', gitf_w=weights).item()
        # unweighted na is identical for both examples here
        self.assertAlmostEqual(weighted, base_loss + 2.0 * NA_UNWEIGHTED * (0.5 + 2.0) / 2,
                               places=3)

    def test_na_layer_prob_weights_score_ratio(self):
        # explicit prob layer under 'example' mode still weights the probability
        current = current_data()
        weights = torch.tensor([0.5, 1.5])
        base = stub_computer(build_base_model('example'))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights).item()
        model = stub_computer(build_lightccf('example', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0,
                                             na_layer='prob'))
        weighted = model.loss(current, reduction='mean', gitf_w=weights).item()
        expected_na = (
            -math.log(0.5 * NA_RATIO + 1e-5) - math.log(1.5 * NA_RATIO + 1e-5)
        ) / 2
        self.assertAlmostEqual(weighted, base_loss + 2.0 * expected_na, places=4)

    def test_na_layer_score_scales_positive_vectors(self):
        # explicit score layer: p_vectors scaled by the weight before scoring
        current = current_data()
        weights = torch.tensor([0.5, 2.0])
        base = stub_computer(build_base_model('margin'))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights).item()
        model = stub_computer(build_lightccf('margin', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0,
                                             na_layer='score'))
        weighted = model.loss(current, reduction='mean', gitf_w=weights).item()
        # user0: pos score exp(0.5), denominator exp(1.5)+exp(0)
        # user1: pos score exp(2), denominator exp(3)+exp(0)
        expected_na = (
            -math.log(math.exp(0.5) / (math.exp(1.5) + 1) + 1e-5)
            - math.log(math.exp(2.0) / (math.exp(3.0) + 1) + 1e-5)
        ) / 2
        self.assertAlmostEqual(weighted, base_loss + 2.0 * expected_na, places=4)

    def test_ssl_margin_both_matches_margin_positive_weights(self):
        # margin_both uses only the positive-weight column for the na term,
        # so with the same positive weights the na contribution matches margin
        current = current_data()
        margin_weights = torch.tensor([0.5, 2.0])
        both_weights = torch.tensor([[0.5, 3.0, 3.0], [2.0, 7.0, 7.0]])
        base_margin = stub_computer(build_base_model('margin'))
        base_both = stub_computer(build_base_model('margin_both'))
        model_margin = stub_computer(
            build_lightccf('margin', ssl_lambda=1.5, reg_lambda=0.0, tau=1.0)
        )
        model_both = stub_computer(
            build_lightccf('margin_both', ssl_lambda=1.5, reg_lambda=0.0, tau=1.0)
        )
        na_margin = (
            model_margin.loss(current, reduction='mean', gitf_w=margin_weights).item()
            - base_margin.loss(current, reduction='mean', gitf_w=margin_weights).item()
        )
        na_both = (
            model_both.loss(current, reduction='mean', gitf_w=both_weights).item()
            - base_both.loss(current, reduction='mean', gitf_w=both_weights).item()
        )
        self.assertAlmostEqual(na_margin, na_both, places=6)

    def test_ssl_example_mode_weights_output_loss(self):
        # example mode: per-example na loss multiplied by the positive weight
        current = current_data()
        weights = torch.tensor([0.5, 2.0])
        base = stub_computer(build_base_model('example'))
        base_loss = base.loss(current, reduction='mean', gitf_w=weights).item()
        model = stub_computer(build_lightccf('example', ssl_lambda=2.0, reg_lambda=0.0, tau=1.0))
        weighted = model.loss(current, reduction='mean', gitf_w=weights).item()
        # unweighted na is identical for both examples here
        self.assertAlmostEqual(weighted, base_loss + 2.0 * NA_UNWEIGHTED * (0.5 + 2.0) / 2,
                               places=3)

    def test_reg_term_formula(self):
        users = torch.tensor([[3.0, 0.0]])
        positives = torch.tensor([[0.0, 4.0]])
        negatives = torch.zeros(2, 2)
        model = build_lightccf()
        reg = model._get_reg_loss(users, positives, negatives)
        # 0.5 * (9/1 + 16/1 + 0/2) = 12.5
        self.assertAlmostEqual(reg.item(), 12.5, places=5)

    def test_reg_lambda_shifts_total_loss(self):
        current = current_data()
        base = stub_computer(build_base_model('margin'))
        base_loss = base.loss(current, reduction='mean', gitf_w=None).item()
        model = stub_computer(build_lightccf('margin', ssl_lambda=0.0, reg_lambda=1.0))
        with_reg = model.loss(current, reduction='mean', gitf_w=None).item()
        # reg = 0.5 * (1/2 + 4/2 + (4+1+1+4+1+1)/4) over the stub embeddings
        expected_reg = (
            0.5 * ALL_USERS.norm(2).pow(2) / 2.0
            + 0.5 * ALL_ITEMS[[0, 3]].norm(2).pow(2) / 2.0
            + 0.5 * ALL_ITEMS[[1, 2, 4, 5]].norm(2).pow(2) / 4.0
        )
        self.assertAlmostEqual(with_reg, base_loss + expected_reg.item(), places=5)


class XavierInitTests(unittest.TestCase):
    def test_embedding_initialized_in_xavier_bounds(self):
        embedding = torch.nn.Embedding(4, 2)
        LightCCF.init_weights(embedding)
        bound = (6.0 / (4 + 2)) ** 0.5
        self.assertTrue((embedding.weight.abs() <= bound).all())
        self.assertGreater(embedding.weight.abs().max().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
