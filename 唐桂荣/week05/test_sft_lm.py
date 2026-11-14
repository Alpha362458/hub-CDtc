"""无需 pytest：在本项目目录运行 python test_sft_lm.py。"""

from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

import sft_lm as sft


class SFTChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.base = sft.load_checkpoint(sft.BASE)
        cls.rows = [
            {"instruction": "选择较大的数：15和28", "output": "较大的数字是28。"},
            {"instruction": "计算1+1", "output": "1+1=2。"},
        ]
        cls.template = dict(sft.DEFAULT_TEMPLATE)
        cls.model, cls.vocab, cls.config = sft.prepare_model(
            cls.base, cls.rows, cls.template, 128)
        cls.model.eval()

    def test_all_pretrained_weights_and_ids_are_preserved(self):
        for char, old_id in self.base["char2idx"].items():
            self.assertEqual(self.vocab[char], old_id)
        state = self.model.state_dict()
        for name, old in self.base["model_state"].items():
            new = state[name]
            if old.shape != new.shape:
                new = new[:old.shape[0]]
            self.assertTrue(torch.equal(new, old), name)
        self.assertIs(self.model.tok.weight, self.model.head.weight)
        self.assertEqual(self.model.pos.num_embeddings, 128)

    def test_model_and_checkpoint_are_inside_this_project(self):
        import inspect
        project = Path(sft.__file__).resolve().parent
        self.assertTrue(sft.BASE.resolve().is_relative_to(project))
        self.assertTrue(Path(inspect.getfile(sft.TransformerLM)).resolve().is_relative_to(project))

    def test_answer_boundary_eos_and_padding(self):
        data = sft.SFTDataset(self.rows, self.vocab, self.template, 128)
        for row, (x, labels) in zip(self.rows, data):
            prompt = sft.prompt_text(row["instruction"], self.template)
            boundary = len(prompt) - 1
            self.assertTrue(all(label == sft.IGNORE for label in labels[:boundary]))
            self.assertEqual(labels[boundary], self.vocab["结"])
            self.assertEqual(x[boundary], self.vocab["："])
            expected = sft.encode(sft.completion_text(row, self.template), self.vocab)
            self.assertEqual(labels[boundary:], expected + [self.vocab[sft.EOS]])
        x, labels = sft.PadBatch(self.vocab[sft.PAD])(data.examples)
        short = min(range(len(data)), key=lambda i: len(data[i][0]))
        tail = len(data[short][0])
        self.assertTrue(torch.all(labels[short, tail:] == sft.IGNORE))
        self.assertTrue(torch.all(x[short, tail:] == self.vocab[sft.PAD]))

    def test_padding_does_not_change_answer_loss(self):
        data = sft.SFTDataset(self.rows, self.vocab, self.template, 128)
        collate = sft.PadBatch(self.vocab[sft.PAD])
        one = sft.evaluate(self.model, DataLoader(data, batch_size=1, collate_fn=collate), "cpu")
        two = sft.evaluate(self.model, DataLoader(data, batch_size=2, collate_fn=collate), "cpu")
        self.assertAlmostEqual(one["answer_loss"], two["answer_loss"], places=5)
        self.assertEqual(one["supervised_tokens"], two["supervised_tokens"])

    def test_causal_mask_blocks_future_information(self):
        x = torch.tensor([sft.encode("问题：计算1+1\n回答：", self.vocab)])
        changed = x.clone()
        changed[0, -1] = self.vocab["结"]
        with torch.no_grad():
            a, b = self.model(x), self.model(changed)
        self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6, rtol=0))
        self.assertFalse(torch.allclose(a[:, -1], b[:, -1]))

    def test_unknown_characters_are_kept_as_unk(self):
        unknown = "\U0010ffff"
        self.assertNotIn(unknown, self.vocab)
        self.assertEqual(sft.encode("1" + unknown + "2", self.vocab),
                         [self.vocab["1"], self.vocab[sft.UNK], self.vocab["2"]])
        row = {"instruction": unknown, "output": "2"}
        data = sft.SFTDataset([row], self.vocab, self.template, 128)
        self.assertEqual(data.unknown_chars, 1)

    def test_long_examples_are_skipped_without_truncating_answers(self):
        long_row = {"instruction": "问" * 200, "output": "回答。"}
        data = sft.SFTDataset(self.rows + [long_row], self.vocab, self.template, 128)
        self.assertEqual(len(data), len(self.rows))
        self.assertEqual(data.skipped, 1)
        with self.assertRaises(ValueError):
            sft.SFTDataset([long_row], self.vocab, self.template, 128)

    def test_checkpoint_reload_preserves_logits_and_weight_tying(self):
        with tempfile.TemporaryDirectory(prefix="week5_sft_test_") as directory:
            path = Path(directory) / "model.pt"
            sft.save_model(path, self.model, self.vocab, self.config,
                           self.template, "pre_sft", {})
            loaded, checkpoint = sft.load_runtime(path, "cpu")
            x = torch.tensor([sft.encode("问题：1+1\n回答：", self.vocab)])
            with torch.no_grad():
                self.assertTrue(torch.equal(self.model(x), loaded(x)))
            self.assertIs(loaded.tok.weight, loaded.head.weight)
            self.assertEqual(checkpoint["template"], self.template)

    def test_eos_stops_generation_without_injecting_answer_prefix(self):
        vocab = self.vocab

        class AlwaysEOS(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros((x.size(0), x.size(1), len(vocab)))
                logits[:, :, vocab[sft.EOS]] = 100
                return logits

        result = sft.generate_text(AlwaysEOS(), vocab, self.config, self.template,
                                   "1+1", "cpu")
        self.assertEqual(result["text"], "")
        self.assertTrue(result["ended_with_eos"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
