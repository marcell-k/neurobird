"""Two layer neural network."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

type Matrix = npt.NDArray[np.float64]
type Vector = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class NeuralNetwork:
    """Input -> tanh -> single linear output."""

    w1: Matrix  # (hidden, inputs)
    b1: Vector  # (hidden,)
    w2: Matrix  # (1, hidden)
    b2: Vector  # (1,)

    @classmethod
    def random(cls, rng: np.random.Generator, input_size: int = 5, hidden_size: int = 6) -> Self:
        """Create a network with normally distributed weights scaled by 1/sqrt(fan_in)."""
        w1 = rng.normal(0.0, 1.0 / np.sqrt(input_size), (hidden_size, input_size))
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_size), (1, hidden_size))
        return cls(w1, np.zeros(hidden_size), w2, np.zeros(1))

    def forward(self, x: Vector) -> float:
        """Run one input vector through the network and return the raw output."""
        hidden = np.tanh(self.w1 @ x + self.b1)
        return float((self.w2 @ hidden + self.b2)[0])

    def mutate(self, rng: np.random.Generator, strength: float) -> NeuralNetwork:
        """Return a new network with Gaussian noise added to every weight and bias."""
        return NeuralNetwork(
            self.w1 + rng.normal(0.0, strength, self.w1.shape),
            self.b1 + rng.normal(0.0, strength, self.b1.shape),
            self.w2 + rng.normal(0.0, strength, self.w2.shape),
            self.b2 + rng.normal(0.0, strength, self.b2.shape),
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a network previously written by `save`."""
        with np.load(path) as data:
            return cls(data["w1"], data["b1"], data["w2"], data["b2"])

    def save(self, path: Path) -> None:
        """Write the weights and biases to a .npz file."""
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
