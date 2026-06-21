from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import lightning as L
import pyrootutils
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run_separation_smoke(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Ejecuta el smoke test de separacion para cada caso definido en la config."""
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False, callbacks=[])

    metric_dict: Dict[str, Any] = {}
    object_dict: Dict[str, Any] = {
        "cfg": cfg,
        "datamodule": datamodule,
        "trainer": trainer,
    }

    for case in cfg.smoke_cases:
        log.info(f"Instantiating model <{cfg.model._target_}> ({case.name})")
        model: LightningModule = hydra.utils.instantiate(
            cfg.model,
            init_identity_mask=case.init_identity_mask,
            case_name=case.case_name,
        )

        trainer.test(model=model, datamodule=datamodule)
        metric_dict.update({k: v for k, v in trainer.callback_metrics.items()})

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="test_separation.yaml")
def main(cfg: DictConfig) -> None:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    extras(cfg)
    run_separation_smoke(cfg)


if __name__ == "__main__":
    main()
