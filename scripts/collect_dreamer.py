"""Collect Yellow trajectories in PokeDreamer's schema (poke/dreamer.py).

Start states: the colour story states (bedroom, Pikachu, the oracle's
milestones) plus every teleport state that scripts/make_start_states.py kept.
Episodes are dealt round-robin over the start states so each one is used,
with the behaviour policy alternating between sticky-random and frontier.

Each worker writes whole episodes to <out>/episodes/; `--pack` then cuts
them into transitions_0001.npz, ... of 10 k steps, as PokeDreamer ships.

  uv run python -m scripts.collect_dreamer --steps 500000 --workers 8 --pack
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROM = "pokeyellow.gbc"
STORY_STATES = ("states/cgb/bedroom.state", "states/cgb/pikachu.state",
                "states/cgb/oracle/rival.state", "states/cgb/oracle/route1.state",
                "states/cgb/oracle/parcel.state", "states/cgb/oracle/pokedex.state",
                "states/cgb/oracle/forest_gate.state", "states/cgb/oracle/forest.state")


def start_states(manifest: Path) -> list[str]:
    states = [path for path in STORY_STATES if Path(path).exists()]
    if manifest.exists():
        rows = json.loads(manifest.read_text())["rows"]
        states += [row["state"] for row in rows if row["ok"]]
    return states


def run_episode(job) -> dict:
    episode_id, state_id, state, policy_id, steps, seed, out_dir = job
    from poke.dreamer import EpisodeBuffer, RewardTracker, act, probe
    from poke.dreamer_policies import make
    from poke.env import PokeEnv
    from poke.symbols import load
    from poke.warp import set_names

    rng = np.random.default_rng(seed)
    env = PokeEnv(ROM, state, color=True)
    started = time.time()
    try:
        env.reset()
        set_names(env, load())
        policy = make(policy_id, rng)
        buffer, rewards = EpisodeBuffer(), RewardTracker()
        fields = probe(env)
        rewards.start(fields)
        for t in range(steps):
            action = policy(fields)
            act(env, action)
            after = probe(env)
            buffer.add(fields, action, rewards.step(after), policy_id, state_id,
                       start=(t == 0))
            fields = after
        arrays = buffer.arrays()
    finally:
        env.close()
    path = Path(out_dir) / "episodes" / f"ep_{episode_id:05d}.npz"
    np.savez_compressed(path, **arrays)
    return {"episode": episode_id, "state": state, "policy": policy_id,
            "steps": steps, "maps": int(len(np.unique(arrays["map_ids"]))),
            "tiles": int(len({tuple(r) for r in np.stack(
                [arrays["map_ids"], arrays["ys"], arrays["xs"]], 1)})),
            "battle_share": float(arrays["in_battles"].mean()),
            "dialog_share": float(arrays["dialog_opens"].mean()),
            "seconds": round(time.time() - started, 1), "file": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=500_000, help="total steps")
    parser.add_argument("--episode-steps", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/pokedreamer_yellow")
    parser.add_argument("--manifest", default="results/start_states_cgb.json")
    parser.add_argument("--pack", action="store_true",
                        help="cut the episodes into 10 k-step chunks afterwards")
    args = parser.parse_args()

    from poke.dreamer import pack
    from poke.dreamer_policies import POLICY_NAMES

    out = Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    states = start_states(Path(args.manifest))
    episodes = -(-args.steps // args.episode_steps)
    # The policy flips on every pass over the states: with about one episode
    # per state, `k % 2` alone would give each state the same policy forever.
    jobs = [(k, k % len(states), states[k % len(states)],
             (k + k // len(states)) % len(POLICY_NAMES),
             args.episode_steps, args.seed * 1_000_003 + k, str(out))
            for k in range(episodes)]
    print(f"{episodes} episodes x {args.episode_steps} steps over {len(states)} "
          f"start states, {args.workers} workers", flush=True)

    rows = []
    started = time.time()
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(run_episode, jobs):
            rows.append(row)
            print(f"ep {row['episode']:4d} {POLICY_NAMES[row['policy']]:13s} "
                  f"maps {row['maps']:2d} tiles {row['tiles']:4d} "
                  f"battle {row['battle_share']:.2f} dialog {row['dialog_share']:.2f} "
                  f"{row['seconds']}s  {Path(row['state']).stem}", flush=True)
    rows.sort(key=lambda row: row["episode"])
    meta = {"rom": ROM, "states": states, "policies": POLICY_NAMES,
            "episode_steps": args.episode_steps, "seed": args.seed,
            "seconds": round(time.time() - started, 1), "episodes": rows}
    (out / "collection.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(rows)} episodes in {meta['seconds']}s", flush=True)

    if args.pack:
        chunks = pack([Path(row["file"]) for row in rows], out)
        print(f"packed {len(chunks)} chunks into {out}", flush=True)


if __name__ == "__main__":
    main()
