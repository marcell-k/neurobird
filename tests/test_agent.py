import random

from neurobird import sim
from neurobird.agent import OBSERVATION_SIZE, observe
from neurobird.config import EASY


def test_observation_matches_declared_size() -> None:
    game = sim.Game(random.Random(0), EASY)
    assert observe(game).shape == (OBSERVATION_SIZE,)
