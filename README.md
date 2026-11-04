# neurobird

Flappy Bird in Pygame, plus a neural network that learns to play it through neuroevolution.

## Setup

```sh
uv sync
```

## Usage

```sh
uv run neurobird                          # play (space to flap)
uv run neurobird --difficulty hard        # harder: speed, gap and spacing ramp up
uv run neurobird-train --generations 100  # curriculum: easy warm-up, then hard
uv run neurobird-train --no-curriculum    # train on one difficulty only
uv run neurobird --ai best.npz            # watch the trained network play
```

Training targets `hard` by default: the first 30% of generations run on easy, then the
rest on hard. Evaluation uses four worker processes by default; `--workers N` changes
that and `--workers 1` disables multiprocessing. Both commands accept
`--difficulty easy|hard`. A hard-trained network also plays easy, but pass
`--difficulty hard` when watching to see the target rules.
