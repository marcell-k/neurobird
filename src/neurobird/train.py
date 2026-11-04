"""Evolve a population of networks to play the game and save the best one."""

import argparse
from pathlib import Path

import numpy as np

from neurobird import sim
from neurobird.agent import run_episode
from neurobird.nn import NeuralNetwork

POPULATION = 100
ELITES = 10
EPISODES = 3  # games per network per generation, on shared seeds
MUTATION_STRENGTH = 0.1


def evaluate(nn: NeuralNetwork, seeds: list[int], config: sim.GameConfig) -> float:
    """Average fitness of a network over the given game seeds."""
    return float(np.mean([run_episode(nn, seed, config) for seed in seeds]))


def train(generations: int, rng: np.random.Generator, config: sim.GameConfig) -> NeuralNetwork:
    """Run the evolution loop and return the best network of the last generation."""
    population = [NeuralNetwork.random(rng) for _ in range(POPULATION)]
    best = population[0]
    for generation in range(generations):
        # Same seeds for everyone in a generation, so the comparison is fair.
        seeds = rng.integers(0, 2**31, EPISODES).tolist()
        scores = [evaluate(nn, seeds, config) for nn in population]
        order = np.argsort(scores)[::-1]
        ranked = [population[i] for i in order]
        best = ranked[0]
        print(  # noqa: T201
            f"gen {generation + 1:4d}  best {scores[order[0]]:9.1f}  "
            f"mean {float(np.mean(scores)):9.1f}"
        )
        elites = ranked[:ELITES]
        children = [
            elites[int(rng.integers(ELITES))].mutate(rng, MUTATION_STRENGTH)
            for _ in range(POPULATION - ELITES)
        ]
        population = [*elites, *children]
    return best


def main() -> None:
    """Command line entry point: train and save the best network."""
    parser = argparse.ArgumentParser(description="Train a neural network to play the game.")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("best.npz"))
    parser.add_argument("--difficulty", choices=sorted(sim.CONFIGS), default="easy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    best = train(args.generations, rng, sim.CONFIGS[args.difficulty])
    best.save(args.output)
    print(f"Saved best network to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
