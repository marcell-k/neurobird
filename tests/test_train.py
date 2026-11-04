from dataclasses import replace

from neurobird.config import DEFAULT_TRAINING, EASY, HARD
from neurobird.train import config_at, warmup_generations


def test_warmup_is_thirty_percent_of_generations() -> None:
    assert warmup_generations(50, HARD, DEFAULT_TRAINING) == 15  # noqa: PLR2004
    assert warmup_generations(10, HARD, DEFAULT_TRAINING) == 3  # noqa: PLR2004


def test_warmup_uses_training_fraction() -> None:
    training = replace(DEFAULT_TRAINING, curriculum_fraction=0.5)
    assert warmup_generations(10, HARD, training) == 5  # noqa: PLR2004


def test_warmup_leaves_at_least_one_target_generation() -> None:
    assert warmup_generations(1, HARD, DEFAULT_TRAINING) == 0
    assert warmup_generations(2, HARD, DEFAULT_TRAINING) == 1


def test_warmup_disabled_when_target_is_easy() -> None:
    assert warmup_generations(50, EASY, DEFAULT_TRAINING) == 0


def test_config_at_switches_from_easy_to_target() -> None:
    assert config_at(14, 15, HARD) == EASY
    assert config_at(15, 15, HARD) == HARD
    assert config_at(49, 15, HARD) == HARD


def test_config_at_without_warmup_stays_on_target() -> None:
    assert config_at(0, 0, HARD) == HARD
