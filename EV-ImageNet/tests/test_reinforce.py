import ast
import pathlib
import textwrap
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


ROOT = pathlib.Path(__file__).resolve().parents[1]
namespace = dict(torch=torch, nn=nn, F=F, Normal=Normal)
tree = ast.parse((ROOT / 'timm/models/retinal/scanpath.py').read_text(encoding='utf-8-sig'))
classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
           and node.name in {'LocationNetwork', 'SRNetwork'}]
exec(compile(ast.Module(body=classes, type_ignores=[]), '<controllers>', 'exec'), namespace)
source = (ROOT / 'train_ev.py').read_text(encoding='utf-8-sig')
start = source.index('                baselines = torch.stack(baselines).transpose(1, 0)')
stop = source.index('                reinforce_weight = 0.1', start)
body = textwrap.dedent(source[start:stop])
function = 'def losses(pre, labels, baselines, log_pi, log_pi_sr):\n'
function += textwrap.indent(body, '    ')
function += '    return loss_baseline, loss_reinforce\n'
exec(compile(function, '<training-loss>', 'exec'), namespace)


class ReinforceTests(unittest.TestCase):
    def test_controller_reward_gradients(self):
        torch.manual_seed(23)
        for name, args, head in [
            ('LocationNetwork', (8, 2, 0.2), 'fc_lt'),
            ('SRNetwork', (8, 1, 2, 0.05), 'fc_s'),
        ]:
            model = namespace[name](*args)
            state = torch.randn(4096, 8, requires_grad=True)
            log_prob, action = model(state)
            self.assertFalse(action.requires_grad)
            reward = action[:, 0]
            advantage = (reward - reward.mean()).detach()
            loss = -(log_prob * advantage).mean()
            loss.backward()
            self.assertLess(getattr(model, head).bias.grad[0].item(), 0)
            self.assertIsNone(state.grad)
            self.assertTrue(torch.isfinite(log_prob).all())

    def test_only_executed_actions_contribute(self):
        for batch in (1, 3):
            baselines = [torch.zeros(batch, requires_grad=True) for _ in range(3)]
            locations = [torch.zeros(batch, requires_grad=True) for _ in range(3)]
            scales = [torch.zeros(batch, requires_grad=True) for _ in range(3)]
            pre = torch.tensor([[3.0, 0.0]]).repeat(batch, 1).requires_grad_()
            labels = torch.zeros(batch, dtype=torch.long)
            baseline_loss, policy_loss = namespace['losses'](
                pre, labels, baselines, locations, scales)
            policy_loss.backward(retain_graph=True)
            self.assertTrue(all(value.grad is None for value in baselines))
            self.assertIsNone(pre.grad)
            for values in (locations, scales):
                self.assertTrue(torch.all(values[0].grad < 0))
                self.assertTrue(torch.all(values[1].grad < 0))
                self.assertTrue(torch.all(values[2].grad == 0))
            baseline_loss.backward()
            self.assertTrue(torch.all(baselines[-1].grad == 0))

    def test_single_observation_has_no_policy_loss(self):
        value = torch.zeros(1, requires_grad=True)
        baseline_loss, policy_loss = namespace['losses'](
            torch.tensor([[1.0, 0.0]]), torch.tensor([0]),
            [value], [value], [value])
        self.assertEqual(baseline_loss.item(), 0.0)
        self.assertEqual(policy_loss.item(), 0.0)
        (baseline_loss + policy_loss).backward()
        self.assertEqual(value.grad.item(), 0.0)


if __name__ == '__main__':
    unittest.main()
