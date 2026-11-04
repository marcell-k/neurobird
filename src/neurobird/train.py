"""Evolve a population of networks to play the game and save the best one."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from neurobird.agent import OBSERVATION_SIZE, run_episode
from neurobird.config import CONFIGS, DEFAULT_TRAINING, EASY, GameConfig, TrainingConfig
from neurobird.nn import NeuralNetwork


def evaluate(nn: NeuralNetwork, seeds: list[int], config: GameConfig) -> float:
    """Average fitness of a network over the given game seeds."""
    return float(np.median([run_episode(nn, seed, config) for seed in seeds]))


def _evaluate_task(task: tuple[NeuralNetwork, list[int], GameConfig]) -> float:
    """Evaluate one network; the entry point used by worker processes."""
    nn, seeds, config = task
    return evaluate(nn, seeds, config)


def _evaluate_all(
    pool: ProcessPoolExecutor | None,
    networks: list[NeuralNetwork],
    seeds: list[int],
    config: GameConfig,
) -> list[float]:
    """Score each network over the same seeds, serially or with a worker pool."""
    if pool is None:
        return [evaluate(nn, seeds, config) for nn in networks]
    tasks = [(nn, seeds, config) for nn in networks]
    return list(pool.map(_evaluate_task, tasks))


def episodes_at(generation: int, generations: int, training: TrainingConfig) -> int:
    """Return the number of episodes per network, growing linearly from start to end."""
    t = generation / max(generations - 1, 1)
    return round(training.episodes_start + (training.episodes_end - training.episodes_start) * t)


def mutation_at(generation: int, generations: int, training: TrainingConfig) -> float:
    """Mutation strength, decaying geometrically from start to end."""
    t = generation / max(generations - 1, 1)
    return training.mutation_start * (training.mutation_end / training.mutation_start) ** t


def warmup_generations(generations: int, final_config: GameConfig, training: TrainingConfig) -> int:
    """Return the number of easy generations before switching to the target difficulty."""
    if final_config == EASY or generations <= 1:
        return 0
    return min(round(generations * training.curriculum_fraction), generations - 1)


def config_at(generation: int, easy_generations: int, final_config: GameConfig) -> GameConfig:
    """Difficulty used for a generation: easy during the warm-up, target afterwards."""
    return EASY if generation < easy_generations else final_config


def _tournament(
    ranked: list[NeuralNetwork], rng: np.random.Generator, training: TrainingConfig
) -> NeuralNetwork:
    """Pick the best of a few random entries. `ranked` is sorted best first."""
    picks = rng.integers(0, len(ranked), training.tournament_size)
    return ranked[int(picks.min())]


def _make_child(
    ranked: list[NeuralNetwork],
    rng: np.random.Generator,
    strength: float,
    training: TrainingConfig,
) -> NeuralNetwork:
    """Select a parent (maybe two, with crossover), then mutate the result."""
    child = _tournament(ranked, rng, training)
    if rng.random() < training.crossover_prob:
        child = child.crossover(_tournament(ranked, rng, training), rng)
    return child.mutate(rng, strength, training.mutation_rate)


def train(
    generations: int,
    rng: np.random.Generator,
    config: GameConfig,
    *,
    training: TrainingConfig = DEFAULT_TRAINING,
) -> NeuralNetwork:
    """Run the evolution loop and return the champion that does best on held-out seeds."""
    if training.workers <= 1:
        return _train(generations, rng, config, training, None)
    max_workers = min(training.workers, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
        return _train(generations, rng, config, training, pool)


def _train(
    generations: int,
    rng: np.random.Generator,
    config: GameConfig,
    training: TrainingConfig,
    pool: ProcessPoolExecutor | None,
) -> NeuralNetwork:
    """Run the evolution loop with an already configured evaluation pool."""
    easy_generations = (
        warmup_generations(generations, config, training) if training.curriculum else 0
    )
    population = [NeuralNetwork.random(rng, OBSERVATION_SIZE) for _ in range(training.population)]
    holdout_seeds = rng.integers(0, 2**31, training.holdout_episodes).tolist()
    hall: list[tuple[float, NeuralNetwork]] = []  # (training score, champion) per generation
    target_name = "easy" if config == EASY else "hard"
    for generation in range(generations):
        if generation == easy_generations and easy_generations > 0:
            print(f"switching to {target_name} at generation {generation + 1}")  # noqa: T201
        # Same seeds for everyone in a generation, so the comparison is fair.
        generation_config = config_at(generation, easy_generations, config)
        episodes = episodes_at(generation, generations, training)
        strength = mutation_at(generation, generations, training)
        seeds = rng.integers(0, 2**31, episodes).tolist()
        scores = _evaluate_all(pool, population, seeds, generation_config)
        order = np.argsort(scores)[::-1]
        ranked = [population[i] for i in order]
        # Only target-difficulty champions are comparable, so easy warm-up scores are skipped.
        if generation >= easy_generations:
            hall.append((scores[order[0]], ranked[0]))
        if generation % 10 == 0:
            phase = "easy" if generation < easy_generations else target_name
            print(  # noqa: T201
                f"gen {generation + 1:4d}  {phase:4s}  eps {episodes:2d}  mut {strength:.3f}  "
                f"best {scores[order[0]]:9.1f}  mean {float(np.mean(scores)):9.1f}"
            )
        children = [
            _make_child(ranked, rng, strength, training)
            for _ in range(training.population - training.elites)
        ]
        population = [*ranked[: training.elites], *children]
    return _pick_champion(hall, holdout_seeds, config, training, pool)


def _pick_champion(
    hall: list[tuple[float, NeuralNetwork]],
    holdout_seeds: list[int],
    config: GameConfig,
    training: TrainingConfig,
    pool: ProcessPoolExecutor | None,
) -> NeuralNetwork:
    """Re-score the best hall-of-fame entries on fixed unseen seeds and return the winner."""
    top = sorted(hall, key=lambda entry: entry[0], reverse=True)[: training.hall_size]
    held_out = _evaluate_all(pool, [nn for _, nn in top], holdout_seeds, config)
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
    parser.add_argument("--difficulty", choices=sorted(CONFIGS), default="hard")
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAINING.curriculum,
        help="train on easy first, then switch to --difficulty",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(DEFAULT_TRAINING.workers, os.cpu_count() or 1),
        help="parallel evaluation processes (1 disables multiprocessing)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    training = replace(DEFAULT_TRAINING, curriculum=args.curriculum, workers=args.workers)
    rng = np.random.default_rng(args.seed)
    best = train(args.generations, rng, CONFIGS[args.difficulty], training=training)
    best.save(args.output)
    print(f"Saved best network to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
