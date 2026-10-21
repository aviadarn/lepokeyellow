"""How many emulator steps per second does this machine do, in parallel?

The encoder runs at ~1,758 frames/s and training at ~403 frames/s, both
measured. The emulator's parallel rate is not, and it is what decides how long
an RL run takes. So: N processes, each stepping its own PokeEnv with random
actions, and the honest total.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from poke.actions import n_actions


def _worker(job):
    rom, state, steps, seed = job
    from poke.env import PokeEnv           # imported here: one PyBoy per process
    rng = random.Random(seed)
    env = PokeEnv(rom, state)
    env.reset()
    started = time.time()
    for _ in range(steps):
        env.step(rng.randrange(n_actions()))
    elapsed = time.time() - started
    env.close()
    return steps / elapsed


def summarise(per_worker_rates, single: float) -> dict:
    if not per_worker_rates:
        raise ValueError("need at least one worker to summarise")
    total = sum(per_worker_rates)
    return {
        "workers": len(per_worker_rates),
        "total_steps_per_s": round(total, 2),
        "per_worker_steps_per_s": round(total / len(per_worker_rates), 2),
        "single_worker_steps_per_s": round(single, 2),
        "efficiency": round(total / (single * len(per_worker_rates)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out", default="results/worker_throughput.json")
    args = parser.parse_args()

    report = {}
    single = _worker((args.rom, args.state, args.steps, 0))
    for count in args.workers:
        jobs = [(args.rom, args.state, args.steps, seed) for seed in range(count)]
        with Pool(count) as pool:
            rates = pool.map(_worker, jobs)
        report[str(count)] = summarise(rates, single)
        print(json.dumps(report[str(count)]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    best = max(report.values(), key=lambda row: row["total_steps_per_s"])
    print(f"best: {best['workers']} workers, "
          f"{best['total_steps_per_s']} steps/s, efficiency {best['efficiency']}")


if __name__ == "__main__":
    main()
