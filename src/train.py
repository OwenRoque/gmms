from typing import Any, Dict, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from pathlib import Path

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, get_metric_value, task_wrapper
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    callbacks = instantiate_callbacks(cfg.get("callbacks")) if cfg.get("callbacks") else []
    loggers = instantiate_loggers(cfg.get("logger")) if cfg.get("logger") else []

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers if loggers else False,
    )

    metric_dict: Dict[str, Any] = {}
    object_dict: Dict[str, Any] = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("test", False):
        log.info("Starting testing!")
        ckpt_path: Optional[str] = None
        if trainer.checkpoint_callback is not None:
            ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
            if ckpt_path == "":
                ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    metric_dict.update({k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in trainer.callback_metrics.items()})

    if cfg.get("optimized_metric"):
        metric_value = get_metric_value(metric_dict, cfg.optimized_metric)
        metric_dict["optimized_metric_value"] = metric_value

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
