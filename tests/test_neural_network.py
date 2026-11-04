import math

import numpy as np
import pytest

from neurobird.nn import NeuralNetwork


def test_forward_matches_hand_computed_output() -> None:
    nn = NeuralNetwork(
        w1=np.eye(2),
        b1=np.zeros(2),
        w2=np.array([[1.0, 2.0]]),
        b2=np.array([0.1]),
    )
    result = nn.forward(np.array([0.5, -0.5]))
    # hidden = [tanh(0.5), -tanh(0.5)]; output = 1*t + 2*(-t) + 0.1
    assert result == pytest.approx(0.1 - math.tanh(0.5))
