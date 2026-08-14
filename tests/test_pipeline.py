"""
Tests for everything that doesn't need a GPU or network access — run these
anywhere, anytime, before ever touching Lightning/Kaggle:

    pytest tests/
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import extract_final_answer, numbers_match
from src.data.preprocess import format_for_training
from src.training.train import find_latest_checkpoint
from src.config import load_config, load_all


class TestMetrics:
    def test_extract_with_delimiter(self):
        assert extract_final_answer("reasoning... #### 72") == "72"

    def test_extract_with_commas(self):
        assert extract_final_answer("total is #### 3,000") == "3000"

    def test_extract_fallback_no_delimiter(self):
        assert extract_final_answer("the answer is 128.") == "128"

    def test_extract_returns_none_when_no_number(self):
        assert extract_final_answer("no numbers here at all") is None

    def test_numbers_match_exact(self):
        assert numbers_match("72", "72") is True

    def test_numbers_match_float_tolerance(self):
        assert numbers_match("45.50", "45.5") is True

    def test_numbers_match_different_values(self):
        assert numbers_match("72", "73") is False

    def test_numbers_match_none_is_false(self):
        assert numbers_match(None, "72") is False


class TestPreprocess:
    def test_format_produces_two_messages(self):
        example = {"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}
        result = format_for_training(example)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_format_preserves_gold_answer(self):
        example = {"question": "test", "answer": "steps... #### 99"}
        result = format_for_training(example)
        assert "#### 99" in result["messages"][1]["content"]


class TestCheckpointResume:
    def test_no_checkpoint_returns_none(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path / "nonexistent")) is None

    def test_picks_highest_step_not_alphabetical(self, tmp_path):
        # Deliberately out of order and mixed digit-length to catch alphabetical-sort bugs
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-300").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        result = find_latest_checkpoint(str(tmp_path))
        assert result.endswith("checkpoint-300")

    def test_double_digit_vs_triple_digit(self, tmp_path):
        # checkpoint-90 should lose to checkpoint-100 numerically,
        # even though '9' > '1' alphabetically
        (tmp_path / "checkpoint-90").mkdir()
        (tmp_path / "checkpoint-100").mkdir()
        result = find_latest_checkpoint(str(tmp_path))
        assert result.endswith("checkpoint-100")


class TestConfig:
    def test_load_model_config(self):
        cfg = load_config("model")
        assert "base_model" in cfg
        assert "lora" in cfg

    def test_load_all_returns_three_configs(self):
        cfg = load_all()
        assert set(cfg.keys()) == {"model", "training", "eval"}

    def test_missing_config_raises_clear_error(self):
        try:
            load_config("nonexistent")
            assert False, "should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "nonexistent" in str(e)
