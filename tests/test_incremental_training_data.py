import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tensorflow_core" / "data"


def read_rows(name):
    return [
        json.loads(line)
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class IncrementalTrainingDataTests(unittest.TestCase):
    def test_core_examples_cover_every_existing_label(self):
        rows = read_rows("ori_core_incremental.jsonl")
        labels = {row["label"] for row in rows}
        self.assertEqual(
            labels,
            {"conversation", "question", "coding", "planning", "troubleshooting", "creative"},
        )
        self.assertTrue(all(row["text"].strip() for row in rows))

    def test_small_examples_use_the_existing_chat_format(self):
        rows = read_rows("ori_small_incremental.jsonl")
        self.assertGreaterEqual(len(rows), 24)
        self.assertTrue(all(" Ori:" in row["text"] for row in rows))

    def test_small_incremental_examples_are_unique(self):
        rows = read_rows("ori_small_incremental.jsonl")
        normalized = [" ".join(row["text"].split()).casefold() for row in rows]
        self.assertEqual(len(normalized), len(set(normalized)))

    def test_incremental_script_reuses_weights_and_tokenizer(self):
        script = (
            ROOT / "tensorflow_core" / "fine_tune_existing.py"
        ).read_text(encoding="utf-8")
        self.assertIn("model.load_weights(required[2])", script)
        self.assertIn("OriTokenizer.load(required[1])", script)
        self.assertIn('"tokenizer_rebuilt": False', script)
        self.assertNotIn("SentencePieceTrainer", script)
        self.assertNotIn("train_tokenizer(", script)

    def test_candidate_is_promoted_only_when_evaluation_improves(self):
        script = (
            ROOT / "tensorflow_core" / "fine_tune_existing.py"
        ).read_text(encoding="utf-8")
        self.assertIn("candidate_loss < baseline_loss", script)
        self.assertIn("candidate_accuracy >= baseline_accuracy", script)


if __name__ == "__main__":
    unittest.main()
