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


def build_model(weight_mode, num_neg=2):
    model = object.__new__(Model)
    model.num_neg = num_neg
    model.gitf_weight_mode = weight_mode
    model._device = torch.device('cpu')
    # users 0, 1 ; items 0..5 with dot products:
    # user 0: item 0 -> 1.0, item 1 -> 0.5, item 2 -> 0.0
    # user 1: item 3 -> 2.0, item 4 -> 1.0, item 5 -> 1.0
    all_users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    all_items = torch.tensor([
        [1.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.0],
        [0.0, 2.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    model.computer = lambda: (all_users, all_items)
    return model


def current_data(num_neg=2):
    batch = 2
    return {
        'user_id': torch.tensor([[0], [1]]),
        'item_id': torch.arange(batch * (num_neg + 1), dtype=torch.long).reshape(
            batch, num_neg + 1
        ),
    }


class GitfWeightModeTests(unittest.TestCase):
    def test_margin_both_scales_positive_and_negative_predictions(self):
        model = build_model('margin_both')
        weights = torch.tensor([
            [2.0, 3.0, 0.5],  # user 0: pos weight 2, neg weights 3, 0.5
            [1.0, 2.0, 4.0],  # user 1: pos weight 1, neg weights 2, 4
        ])
        pos_pred = torch.tensor([1.0, 2.0])
        neg_pred = torch.tensor([[0.5, 0.0], [1.0, 1.0]])
        expected_margin = torch.stack([
            2.0 * pos_pred[0] - weights[0, 1:] * neg_pred[0],
            1.0 * pos_pred[1] - weights[1, 1:] * neg_pred[1],
        ])

        loss = model.loss(current_data(), reduction='mean', gitf_w=weights)

        expected = torch.nn.functional.softplus(expected_margin).mean(dim=1).mean()
        self.assertTrue(torch.isclose(loss, expected, atol=1e-6))

    def test_margin_both_matches_margin_mode_when_all_weights_are_equal(self):
        weight = 1.7
        losses = {}
        for mode, weights in (
            ('margin', torch.tensor([weight, weight])),
            ('margin_both', torch.full((2, 3), weight)),
        ):
            model = build_model(mode)
            losses[mode] = model.loss(
                current_data(), reduction='mean', gitf_w=weights
            ).item()
        self.assertAlmostEqual(losses['margin'], losses['margin_both'], places=6)

    def test_margin_and_example_modes_unchanged(self):
        weights = torch.tensor([0.5, 2.0])
        pos_pred = torch.tensor([1.0, 2.0])
        neg_pred = torch.tensor([[0.5, 0.0], [1.0, 1.0]])
        margin = pos_pred[:, None] - neg_pred

        margin_model = build_model('margin')
        margin_loss = margin_model.loss(
            current_data(), reduction='mean', gitf_w=weights
        )
        expected_margin = torch.nn.functional.softplus(margin * weights[:, None])
        expected_margin = expected_margin.mean(dim=1).mean()
        self.assertTrue(torch.isclose(margin_loss, expected_margin, atol=1e-6))

        example_model = build_model('example')
        example_loss = example_model.loss(
            current_data(), reduction='mean', gitf_w=weights
        )
        expected_example = (
            torch.nn.functional.softplus(margin) * weights[:, None]
        ).mean(dim=1).mean()
        self.assertTrue(torch.isclose(example_loss, expected_example, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
