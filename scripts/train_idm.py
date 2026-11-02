"""Train the inverse dynamics model (poke/idm.py) on data/synth.

Episodes are split by MAP: every teleport start state whose map falls in a
held-out set goes to validation, so the score says how the model reads
places it has never seen -- which is most of the longplay. Story states
(bedroom, lab, Route 1 ...) always train.

The number that matters is balanced accuracy where it matters: on frames
where the exact counterfactual says a button would change the screen, the
mean of (a) presses caught -- held 1, predicted 1 -- and (b) releases kept --
held 0, predicted 0. Both halves are needed: on those frames the bit is 0
92.6 % of the time (pressing A where nobody did would have advanced the
text), so plain accuracy credits a model that predicts nothing with 0.926.
Balanced, that model scores 0.5. Everywhere else the held bit is the prior.

  uv run python -m scripts.train_idm --steps 8000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from poke.idm import IDM, cut_loss, loss, onset_loss, onset_targets

WINDOW = 32


def episode_rows(synth: Path) -> list[dict]:
    """(file, state) per finished episode.

    collection.json is written when a run ends; while one is still going,
    the state of episode k is recovered the way synth_collect assigned it,
    states[k % len(states)], so training can start on what exists.
    """
    meta = synth / "collection.json"
    # Every other batch (power-on, bump-A, ...) writes collection_<tag>.json.
    extra = [row for path in sorted(synth.glob("collection_*.json"))
             for row in json.loads(path.read_text())["episodes"]]
    if meta.exists():
        return json.loads(meta.read_text())["episodes"] + extra
    from scripts.collect_dreamer import start_states

    states = start_states(Path("results/start_states_cgb.json"))
    rows = []
    for path in sorted(synth.glob("ep_*.npz")):
        k = int(path.stem.split("_")[1])
        if k < 10_000:
            rows.append({"file": str(path), "state": states[k % len(states)]})
    return rows + extra


def memmaps(synth: Path) -> list[dict]:
    """Episode arrays as uncompressed .npy memmaps (built once from the npz)."""
    cache = synth / "mm"
    cache.mkdir(exist_ok=True)
    episodes = []
    for row in episode_rows(synth):
        stem = Path(row["file"]).stem
        paths = {key: cache / f"{stem}.{key}.npy" for key in ("frames", "buttons", "mattered")}
        if not all(p.exists() for p in paths.values()):
            with np.load(row["file"]) as data:
                for key, path in paths.items():
                    np.save(path, data[key])
        episodes.append({"state": row["state"], "stem": stem,
                         **{key: np.load(path, mmap_mode="r") for key, path in paths.items()}})
    return episodes


def map_of(state: str) -> str | None:
    """Map label of a teleport start state; None for story states."""
    match = re.match(r"(.+)_w\d+$", Path(state).stem)
    return match.group(1) if "teleport" in state and match else None


def split(episodes: list[dict], share: float, seed: int):
    maps = sorted({m for e in episodes if (m := map_of(e["state"]))})
    rng = np.random.default_rng(seed)
    held_out = {str(m) for m in rng.choice(maps, max(1, int(len(maps) * share)), replace=False)}
    train = [e for e in episodes if map_of(e["state"]) not in held_out]
    val = [e for e in episodes if map_of(e["state"]) in held_out]
    return train, val, sorted(held_out)


def _window(episodes, rng):
    e = episodes[rng.integers(len(episodes))]
    start = int(rng.integers(0, len(e["buttons"]) - WINDOW))
    frames = np.asarray(e["frames"][start:start + WINDOW + 1])
    bits = np.asarray(e["buttons"][start:start + WINDOW + 1])
    matter = np.asarray(e["mattered"][start:start + WINDOW + 1])
    return frames, bits, matter


def batch(episodes, size, rng, splice: float = 0.0):
    """`splice` of the windows are two windows joined at a random frame, with
    the join labelled as a cut (and its mattered mask unknown)."""
    frames, held, matter, cuts, previous = [], [], [], [], []
    for _ in range(size):
        f, bits, m = _window(episodes, rng)
        cut = np.zeros(WINDOW + 1, np.uint8)
        if rng.random() < splice:
            g, other_bits, other_m = _window(episodes, rng)
            at = int(rng.integers(2, WINDOW))
            f, bits, m = f.copy(), bits.copy(), m.copy()
            f[at:], bits[at:], m[at:] = g[at:], other_bits[at:], other_m[at:]
            m[at] = 255
            cut[at] = 1
        frames.append(f)
        previous.append((bits[0] >> np.arange(8)) & 1)
        held.append((bits[1:, None] >> np.arange(8)) & 1)
        matter.append(m[1:])
        cuts.append(cut[1:])
    # uint8 all the way to the device: as int64 on the CPU a batch of 32
    # windows is ~200 MB of copying per step.
    return (torch.from_numpy(np.stack(frames)),
            torch.from_numpy(np.stack(held).astype(np.uint8)),
            torch.from_numpy(np.stack(matter)),
            torch.from_numpy(np.stack(cuts)),
            torch.from_numpy(np.stack(previous).astype(np.uint8)))


@torch.no_grad()
def evaluate(model, episodes, device, windows: int = 400, seed: int = 0) -> dict:
    """Centre-frame metrics over fixed random windows of `episodes`."""
    model.eval()
    rng = np.random.default_rng(seed)
    centre = slice(WINDOW // 4, 3 * WINDOW // 4)
    pred, true, matter, mprob, cut_p, cut_t, on_p, on_t = [], [], [], [], [], [], [], []
    for _ in range(windows // 32):
        frames, held, matt, cuts, prev = batch(episodes, 32, rng, splice=0.5)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "mps"):
            h, m, c, o = model(frames.to(device).long())
        target, _ = onset_targets(held, prev)
        on_p.append(torch.sigmoid(o.float())[:, centre].cpu().numpy().reshape(-1, 8))
        on_t.append(target[:, centre].numpy().reshape(-1, 8))
        cut_p.append(torch.sigmoid(c.float()).cpu().numpy().ravel())
        cut_t.append(cuts.numpy().ravel())
        pred.append(torch.sigmoid(h.float())[:, centre].cpu().numpy().reshape(-1, 8))
        mprob.append(torch.sigmoid(m.float())[:, centre].cpu().numpy().reshape(-1, 8))
        true.append(held[:, centre].numpy().reshape(-1, 8))
        matter.append(matt[:, centre].numpy().reshape(-1, 8))
    model.train()
    pred, true, matter, mprob = map(np.concatenate, (pred, true, matter, mprob))
    cut_p, cut_t = np.concatenate(cut_p), np.concatenate(cut_t)
    on_p, on_t = np.concatenate(on_p), np.concatenate(on_t)
    hard = pred > 0.5
    known = matter == 1
    pressed, released = known & (true == 1), known & (true == 0)
    recall = float(hard[pressed].mean()) if pressed.any() else None
    keep = float((~hard)[released].mean()) if released.any() else None
    out = {
        "frames": int(len(true)),
        "accuracy_all": float((hard == true).mean()),
        "press_recall_where_mattered": recall,
        "release_kept_where_mattered": keep,
        "balanced_where_mattered": (recall + keep) / 2 if None not in (recall, keep) else None,
        "mattered_pairs": int(known.sum()),
        "mattered_presses": int(pressed.sum()),
    }
    labelled = matter != 255
    if labelled.any():
        out["mattered_head_accuracy"] = float(((mprob > 0.5) == (matter == 1))[labelled].mean())
    per = {}
    for k in range(8):
        tp = int((hard[:, k] & (true[:, k] == 1)).sum())
        fp = int((hard[:, k] & (true[:, k] == 0)).sum())
        fn = int((~hard[:, k] & (true[:, k] == 1)).sum())
        per[k] = round(2 * tp / max(1, 2 * tp + fp + fn), 3)
    out["f1_per_button"] = per
    onset_f1 = {}
    for k in range(8):
        # within +-1 frame, on the flattened centre frames (windows rarely split a tap)
        p_on = on_p[:, k] > 0.5
        t_on = on_t[:, k] > 0.5
        near_t = t_on | np.roll(t_on, 1) | np.roll(t_on, -1)
        near_p = p_on | np.roll(p_on, 1) | np.roll(p_on, -1)
        tp_r = int((t_on & near_p).sum()); tp_p = int((p_on & near_t).sum())
        rec = tp_r / max(1, int(t_on.sum())); prec = tp_p / max(1, int(p_on.sum()))
        onset_f1[k] = round(2 * rec * prec / max(1e-9, rec + prec), 3)
    out["onset_f1_pm1"] = onset_f1
    flagged = cut_p > 0.5
    out["cut_recall"] = float(flagged[cut_t == 1].mean())
    out["cut_false_per_10k_frames"] = float(flagged[cut_t == 0].mean() * 1e4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth", default="data/synth")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-share", type=float, default=0.12)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--splice", type=float, default=0.2,
                        help="share of training windows spliced from two places (cut head)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="use only episodes numbered below this (collection may be running)")
    parser.add_argument("--init", default=None,
                        help="start from this checkpoint's weights (fine-tune on new data)")
    parser.add_argument("--out", default="checkpoints/idm")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    episodes = memmaps(Path(args.synth))
    if args.max_episodes is not None:
        episodes = [e for e in episodes if int(e["stem"].split("_")[1]) < args.max_episodes]
    train, val, held_out = split(episodes, args.val_share, args.seed)
    print(f"{len(train)} train / {len(val)} val episodes; held-out maps: {held_out}",
          flush=True)

    # Inverse held frequency per button over the training episodes, capped.
    bits = np.concatenate([np.asarray(e["buttons"]) for e in train])
    share = np.array([((bits >> k) & 1).mean() for k in range(8)])
    pos_weight = torch.tensor(np.clip((1 - share) / np.maximum(share, 1e-6), 1.0, args.max_pos_weight),
                              dtype=torch.float32, device=device)
    print("held share", share.round(4).tolist(), "pos_weight", pos_weight.cpu().numpy().round(1).tolist(),
          flush=True)
    model = IDM(window=WINDOW).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        print(f"initialised from {args.init}", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    warmup = max(1, args.steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda step: min(
        (step + 1) / warmup, 0.5 * (1 + np.cos(np.pi * min(1.0, step / args.steps)))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    history, best = [], -1.0
    started = time.time()
    for step in range(1, args.steps + 1):
        frames, held, matter, cuts, prev = batch(train, args.batch, rng, splice=args.splice)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "mps"):
            h, m, c, o = model(frames.to(device).long())
        held, prev = held.to(device), prev.to(device)
        held_loss, matter_loss = loss(h.float(), m.float(), held, matter.to(device),
                                      pos_weight=pos_weight)
        splice_loss = cut_loss(c.float(), cuts.to(device))
        target, weight = onset_targets(held, prev)
        press_loss = onset_loss(o.float(), target, weight)
        (held_loss + matter_loss + splice_loss + press_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        schedule.step()
        if step % 100 == 0:
            print(f"step {step} held {held_loss.item():.4f} mattered {matter_loss.item():.4f} "
                  f"cut {splice_loss.item():.4f} onset {press_loss.item():.4f} "
                  f"{time.time() - started:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "val": evaluate(model, val, device),
                   "train": evaluate(model, train, device, windows=200)}
            history.append(row)
            score = row["val"]["balanced_where_mattered"] or 0.0
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "window": WINDOW, "step": step,
                        "held_out_maps": held_out}, out / "last.pt")
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "window": WINDOW, "step": step,
                            "held_out_maps": held_out}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(
        {"held_out_maps": held_out, "history": history, "args": vars(args)}, indent=1))
    print(f"best val balanced accuracy where mattered {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
