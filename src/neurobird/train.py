"""Evolve a population of networks to play the game and save the best one."""

import argparse
from pathlib import Path

import numpy as np

from neurobird import sim
from neurobird.agent import run_episode
from neurobird.nn import NeuralNetwork

POPULATION = 100
ELITES = 10
EPISODES_START = 4  # games per network in the first generation, on shared seeds
EPISODES_END = 12  # games per network in the last generation
HALL_SIZE = 20  # how many per-generation champions get the final held-out re-evaluation
HOLDOUT_EPISODES = 50  # fixed seeds used only to pick the network that gets saved
MUTATION_STRENGTH = 0.1


def evaluate(nn: NeuralNetwork, seeds: list[int], config: sim.GameConfig) -> float:
    """Average fitness of a network over the given game seeds."""
    return float(np.median([run_episode(nn, seed, config) for seed in seeds]))


def episodes_at(generation: int, generations: int) -> int:
    """Return the number of episodes per network, growing linearly from START to END."""
    t = generation / max(generations - 1, 1)
    return round(EPISODES_START + (EPISODES_END - EPISODES_START) * t)


def train(generations: int, rng: np.random.Generator, config: sim.GameConfig) -> NeuralNetwork:
    """Run the evolution loop and return the champion that does best on held-out seeds."""
    population = [NeuralNetwork.random(rng) for _ in range(POPULATION)]
    holdout_seeds = rng.integers(0, 2**31, HOLDOUT_EPISODES).tolist()
    hall: list[tuple[float, NeuralNetwork]] = []  # (training score, champion) per generation
    for generation in range(generations):
        # Same seeds for everyone in a generation, so the comparison is fair.
        episodes = episodes_at(generation, generations)
        seeds = rng.integers(0, 2**31, episodes).tolist()
        scores = [evaluate(nn, seeds, config) for nn in population]
        order = np.argsort(scores)[::-1]
        ranked = [population[i] for i in order]
        hall.append((scores[order[0]], ranked[0]))
        if generation % 10 == 0:
            print(  # noqa: T201
                f"gen {generation + 1:4d}  eps {episodes:2d}  best {scores[order[0]]:9.1f}  "
                f"mean {float(np.mean(scores)):9.1f}"
            )
        elites = ranked[:ELITES]
        children = [
            elites[int(rng.integers(ELITES))].mutate(rng, MUTATION_STRENGTH)
            for _ in range(POPULATION - ELITES)
        ]
        population = [*elites, *children]
    return _pick_champion(hall, holdout_seeds, config)


def _pick_champion(
    hall: list[tuple[float, NeuralNetwork]], holdout_seeds: list[int], config: sim.GameConfig
) -> NeuralNetwork:
    """Re-score the best hall-of-fame entries on fixed unseen seeds and return the winner."""
    top = sorted(hall, key=lambda entry: entry[0], reverse=True)[:HALL_SIZE]
    held_out = [evaluate(nn, holdout_seeds, config) for _, nn in top]
    winner = int(np.argmax(held_out))
    print(  # noqa: T201
        f"hall of fame: {len(top)} candidates, "
        f"held-out best {held_out[winner]:.1f} (training score {top[winner][0]:.1f})"
    )
    return top[winner][1]


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
