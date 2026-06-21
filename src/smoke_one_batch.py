"""Smoke test: 1 batch del pipeline Slakh on-the-fly + forward + 1 training step.

Uso:
    conda activate gmss
    cd /home/hadoop1/Documentos/gmss
    python src/smoke_one_batch.py

Requiere GPU (RTX 4070). No ejecuta trainer.fit completo.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import pyrootutils
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def run_smoke(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    if not torch.cuda.is_available():
        log.warning("CUDA no disponible; el smoke seguira en CPU (mas lento).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dm = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup("fit")

    log.info(
        f"Dataset sizes — train: {len(dm.data_train)}, "
        f"val: {len(dm.data_val)}, test: {len(dm.data_test)}"
    )

    log.info("Loading one training batch (on-the-fly generation)...")
    batch = next(iter(dm.train_dataloader()))
    log.info(f"Batch shapes: {{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in batch.items())}}}")

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = instantiate(cfg.model)
    model = model.to(device)
    model.train()

    mixture = batch["mixture"].to(device)
    reference = batch["reference"].to(device)
    target = batch["target"].to(device)

    log.info("Forward pass...")
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        separated = model(mixture, reference, output_length=mixture.shape[-1])
    log.info(f"Separated shape: {tuple(separated.shape)}")

    log.info("Training step (1 batch)...")
    batch_gpu = {
        "mixture": mixture,
        "reference": reference,
        "target": target,
    }
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        loss = model.training_step(batch_gpu, 0)
    log.info(f"OK — loss={float(loss):.4f}")
    log.info("Smoke test completado sin errores.")


@hydra.main(version_base="1.3", config_path="../configs", config_name="smoke_one_batch.yaml")
def main(cfg: DictConfig) -> None:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    run_smoke(cfg)


if __name__ == "__main__":
    main()
