"""检查未来信息泄漏、目标对齐及填充边界。python -B test_model.py"""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model import TransformerLM
from prepare_data import TokenBlocks, clean_line, split_lines


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.config = {"vocab_size": 32, "block_size": 16, "d_model": 16,
                       "n_heads": 4, "n_layers": 2, "d_ff": 64, "dropout": 0.1}
        self.model = TransformerLM(self.config).eval()

    def test_causal_no_future_leakage(self):
        a = torch.randint(1, 32, (2, 12))
        b = a.clone()
        b[:, 7:] = torch.randint(1, 32, (2, 5))
        with torch.inference_mode():
            self.assertTrue(torch.allclose(self.model(a)[:, :7], self.model(b)[:, :7], atol=1e-6))
            self.assertTrue(torch.allclose(self.model(a)[:, :7], self.model(a[:, :7]), atol=1e-6))

    def test_padding_cannot_affect_prefix(self):
        ids = torch.tensor([[2, 4, 9, 6]])
        padded = F.pad(ids, (0, 6))
        with torch.inference_mode():
            self.assertTrue(torch.allclose(self.model(ids), self.model(padded)[:, :4], atol=1e-6))

    def test_targets_shift_and_cover_every_token(self):
        for n in (2, 16, 17, 18, 32, 33, 34):
            tokens = np.arange(1, n + 1, dtype=np.uint16)
            ds = TokenBlocks(tokens, 16)
            targets = []
            for x, y in ds:
                mask = y.ne(-100)
                self.assertTrue(torch.equal(y[mask], x[mask] + 1))
                targets.extend(y[mask].tolist())
            self.assertEqual(targets, tokens[1:].tolist())

    def test_finite_gradients_and_weight_tying(self):
        self.model.train()
        x = torch.randint(1, 32, (2, 10))
        loss = F.cross_entropy(self.model(x).reshape(-1, 32), torch.randint(1, 32, (20,)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        self.assertIs(self.model.head.weight, self.model.token_embedding.weight)

    def test_duplicate_normalization_precedes_split(self):
        self.assertEqual(clean_line("　ＡＢＣ  中文 \t句子  "), "ABC 中文 句子")
        raw = "\n".join([f"句子{i}" for i in range(100)] + ["句子1"]*10 + ["  句子2　", ""])
        config = {"seed": 42, "train_ratio": .9, "val_ratio": .05}
        splits, stats = split_lines(raw, config)
        self.assertEqual(stats["unique_lines"], 100)
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertEqual(splits, split_lines(raw, config)[0])

    def test_reject_context_overflow(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones((1, 17), dtype=torch.long))


if __name__ == "__main__":
    unittest.main(verbosity=2)
