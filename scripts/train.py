import torch as t
import einops
from poke.coord_head import CoordHead, coord_loss
from poke.sigreg import SIGReg
from poke.jepa import JEPA
from poke.dataset import PokeDataset
from poke.splits import episode_split
from poke.goals import default_device
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path

"""Train the JEPA. Adapted from stmonty/lePokeRed.

Changed: MPS is a device candidate (upstream picks cuda or cpu, so training on
an Apple box silently lands on CPU), DefaultConfig's fields are exposed as
flags so the dataset and checkpoint directory can move without editing code,
and `--coord-weight` adds the auxiliary coordinate loss from poke.coord_head.

That last one is not a tweak. With `L_pred + lambda L_reg` alone the predictor
reaches an excellent embedding MSE and is still WORSE than assuming the player
never moved at saying where the player will be -- 2.42 tiles against 0.69, held
out (results/rollout_tiles.json) -- so nothing downstream can plan a route.
Nothing in the objective ever asked the prediction to keep a tile legible.
"""


@dataclass
class DefaultConfig:
    data_path: str = "episodes/poke.h5"
    num_frames: int = 4
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    regularizer_weight: float = 0.1
    coord_weight: float = 0.05
    seed: int = 42

def loss_terms(model: JEPA, reg: SIGReg, batch: dict[str, t.Tensor], regularizer_weight: float = 0.1, head: CoordHead | None = None, coord_weight: float = 0.0) -> tuple[t.Tensor, ...]:
    predictions, targets, embeddings = model(batch["frames"], batch["actions"])
    prediction_loss = t.nn.functional.mse_loss(predictions, targets)
    temporal_embeddings = einops.rearrange(embeddings, "b t d -> t b d")
    reg_loss = reg(temporal_embeddings)

    # This represents: L_pred + λL_reg (and yes I had to find a lambda character on Google)
    total_loss = prediction_loss + (regularizer_weight * reg_loss)

    coord_term = prediction_loss.new_zeros(())
    coord_tiles = prediction_loss.new_zeros(())
    if head is not None and coord_weight > 0 and "location" in batch:
        coord_term, coord_tiles = coord_loss(
            head, predictions, targets, batch["location"]
        )
        total_loss = total_loss + coord_weight * coord_term

    return total_loss, prediction_loss, reg_loss, coord_tiles


def optimization_step(model: JEPA, reg: SIGReg, optimizer: Optimizer, batch: dict[str, t.Tensor], regularizer_weight: float = 0.1, head: CoordHead | None = None, coord_weight: float = 0.0) -> tuple[t.Tensor, ...]:
    model.train()
    if head is not None:
        head.train()

    optimizer.zero_grad(set_to_none=True)
    losses = loss_terms(model, reg, batch, regularizer_weight, head, coord_weight)

    # Calculate gradients
    losses[0].backward()

    optimizer.step()

    return tuple(value.detach() for value in losses)

def make_loaders(data_path: str, num_frames: int = 4, batch_size: int = 16, num_workers: int = 0, val_fraction: float = 0.1, seed: int = 42) -> tuple[DataLoader, DataLoader]:
    train_episodes, val_episodes = episode_split(data_path, val_fraction=val_fraction, seed=seed)

    train_dataset = PokeDataset(data_path, num_frames=num_frames, episode_indices=train_episodes)
    val_dataset = PokeDataset(data_path, num_frames=num_frames, episode_indices=val_episodes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=num_workers, pin_memory=t.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=num_workers, pin_memory=t.cuda.is_available())

    return (train_loader, val_loader)


@t.no_grad()
def validate(model: JEPA, reg: SIGReg, loader: DataLoader, device: t.device, regularizer_weight: float = 0.1, head: CoordHead | None = None, coord_weight: float = 0.0) -> tuple[float, ...]:
    model.eval()
    if head is not None:
        head.eval()
    loss_sums = t.zeros(4, device=device)
    number_of_samples = 0

    for batch in tqdm(loader, desc="validation", unit="batch", leave=False):
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }

        losses = loss_terms(model, reg, batch, regularizer_weight, head, coord_weight)
        batch_size = batch["frames"].size(0)
        loss_sums += t.stack(losses) * batch_size
        number_of_samples += batch_size

    averages = loss_sums / number_of_samples
    return tuple(averages.cpu().tolist())

def train_one_epoch(
    model: JEPA,
    reg: SIGReg,
    optimizer: Optimizer,
    loader: DataLoader,
    device: t.device,
    regularizer_weight: float = 0.1,
    head: CoordHead | None = None,
    coord_weight: float = 0.0,
) -> tuple[float, ...]:
    model.train()

    loss_sums = t.zeros(4, device=device)
    number_of_samples = 0

    progress = tqdm(loader, desc="training", unit="batch")

    for batch_index, batch in enumerate(progress, start=1):
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }

        losses = optimization_step(model, reg, optimizer, batch,
                                   regularizer_weight, head, coord_weight)

        batch_size = batch["frames"].size(0)

        loss_sums += t.stack(losses) * batch_size
        number_of_samples += batch_size

        if batch_index % 10 == 0 or batch_index == len(loader):
            running_averages = loss_sums / number_of_samples
            progress.set_postfix(
                total=f"{running_averages[0].item():.4f}",
                pred=f"{running_averages[1].item():.4f}",
                reg=f"{running_averages[2].item():.4f}",
                tiles=f"{running_averages[3].item():.2f}",
            )

    averages = loss_sums / number_of_samples
    return tuple(averages.cpu().tolist())

def save_checkpoint(
    path: Path,
    model: JEPA,
    reg: SIGReg,
    optimizer: Optimizer,
    config: DefaultConfig,
    epoch: int,
    global_step: int,
    train_losses: tuple[float, ...],
    val_losses: tuple[float, ...],
    head: CoordHead | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    t.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(config),
            "jepa": model.state_dict(),
            "coord_head": None if head is None else head.state_dict(),
            "sigreg": reg.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        temporary_path,
    )
    temporary_path.replace(path)

def parse_config() -> tuple[DefaultConfig, Path, str]:
    import argparse

    defaults = DefaultConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, value in asdict(defaults).items():
        parser.add_argument(
            f"--{field_name.replace('_', '-')}",
            type=type(value),
            default=value,
        )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    # Upstream picks the best checkpoint on total validation loss, which is
    # pred + 0.1 * SIGReg. Measured on the first real run, SIGReg dominates it
    # completely: val total was 0.7200 with reg 7.0741, so the regulariser is
    # 0.707 of 0.720. Epochs 3 and 4 had strictly better val PREDICTION
    # (0.0103, 0.0097 vs 0.0125) and were both rejected because reg ticked up.
    # The checkpoint you plan with is then selected on the anti-collapse
    # penalty rather than on how well it predicts, which is the wrong criterion
    # for a model whose only job is to predict. Default changed to "pred".
    parser.add_argument("--select-on", default="pred",
                        choices=("pred", "total", "tiles"),
                        help="validation metric that decides the best "
                             "checkpoint. 'tiles' selects on how well the "
                             "PREDICTED embedding places the player, which is "
                             "what a planner reads and what embedding MSE "
                             "turned out not to measure")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()
    fields = {name: getattr(args, name) for name in asdict(defaults)}
    return (DefaultConfig(**fields), Path(args.checkpoint_dir), args.device,
            args.select_on)


def main() -> None:
    config, checkpoint_dir, device_name, select_on = parse_config()
    t.manual_seed(config.seed)
    device = t.device(device_name)
    train_loader, val_loader = make_loaders(data_path=config.data_path, num_frames=config.num_frames, batch_size=config.batch_size, seed=config.seed)

    model = JEPA(num_frames=config.num_frames).to(device)
    reg = SIGReg().to(device)
    head = CoordHead(dim=model.dim).to(device) if config.coord_weight > 0 else None

    parameters = list(model.parameters())
    if head is not None:
        parameters += list(head.parameters())
    optimizer = t.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    print(f"device: {device}")
    print(f"train windows: {len(train_loader.dataset)}")
    print(f"validation windows: {len(val_loader.dataset)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"selecting best checkpoint on validation {select_on}")
    if head is None:
        print("no coordinate loss (--coord-weight 0)")
    else:
        print(f"coordinate loss weight {config.coord_weight}")

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_losses = train_one_epoch(
            model,
            reg,
            optimizer,
            train_loader,
            device,
            config.regularizer_weight,
            head,
            config.coord_weight,
        )

        val_losses = validate(
            model,
            reg,
            val_loader,
            device,
            config.regularizer_weight,
            head,
            config.coord_weight,
        )

        print(
            f"epoch {epoch + 1:02d} | "
            f"train total={train_losses[0]:.4f} "
            f"pred={train_losses[1]:.4f} "
            f"reg={train_losses[2]:.4f} | "
            f"val total={val_losses[0]:.4f} "
            f"pred={val_losses[1]:.4f} "
            f"reg={val_losses[2]:.4f} "
            f"tiles={val_losses[3]:.2f}"
        )

        completed_epoch = epoch + 1
        global_step = completed_epoch * len(train_loader)

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            reg,
            optimizer,
            config,
            completed_epoch,
            global_step,
            train_losses,
            val_losses,
            head,
        )

        # val_losses is (total, pred, reg, tiles).
        selected = {"total": val_losses[0], "pred": val_losses[1],
                    "tiles": val_losses[3]}[select_on]
        if selected < best_val_loss:
            best_val_loss = selected
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                reg,
                optimizer,
                config,
                completed_epoch,
                global_step,
                train_losses,
                val_losses,
                head,
            )
            print(f"saved new best checkpoint: {checkpoint_dir / 'best.pt'}")

if __name__ == "__main__":
    main()
