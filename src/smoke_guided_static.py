"""Smoke: entrena GuidedSeparator con WAV estaticos en GPU (sin on-the-fly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log.warning("CUDA no disponible; usando CPU.")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.get("train", True):
        log.info("trainer.fit() — WAV estaticos, sin FluidSynth")
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("test", False):
        ckpt = None
        if trainer.checkpoint_callback is not None:
            ckpt = getattr(trainer.checkpoint_callback, "best_model_path", None)
            if ckpt == "":
                ckpt = None
        log.info("trainer.test()")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt)

    metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
               for k, v in trainer.callback_metrics.items()}
    return metrics, {"cfg": cfg, "trainer": trainer}


@hydra.main(version_base="1.3", config_path="../configs", config_name="smoke_guided_static.yaml")
def main(cfg: DictConfig) -> None:
    """Ejecutar con --config-name=smoke_guided_minimal para prueba mas rapida."""
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    extras(cfg)
    metrics, _ = run(cfg)
    log.info(f"Metricas finales: {metrics}")


if __name__ == "__main__":
    main()
