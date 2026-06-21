"""Entry point de inferencia guiada (mixture + reference -> separated)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
import pyrootutils
import torch
import torchaudio
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.audio_loader import AudioLoader
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded checkpoint from {checkpoint_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    extras(cfg)

    import hydra.utils
    from src.models.guided_separation_module import GuidedSeparationLitModule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model: GuidedSeparationLitModule = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.eval()

    if cfg.get("checkpoint_path"):
        _load_checkpoint(model, Path(cfg.checkpoint_path))

    feature = model.hparams.feature
    loader = AudioLoader(
        sample_rate=feature.sample_rate,
        mono=feature.mono,
        normalize=True,
    )

    mixture = loader.load(cfg.mixture_path)
    reference = loader.load(cfg.reference_path)

    if cfg.get("segment_seconds"):
        seg = int(cfg.segment_seconds * feature.sample_rate)
        mixture = mixture[..., :seg]
        reference = reference[..., :seg]

    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0)
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)

    mixture = mixture.unsqueeze(0).to(device)
    reference = reference.unsqueeze(0).to(device)

    with torch.no_grad():
        separated = model(mixture, reference, output_length=mixture.shape[-1])

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if separated.ndim == 2:
        separated = separated.unsqueeze(1)
    torchaudio.save(str(out_path), separated.squeeze(0).cpu(), feature.sample_rate)
    log.info(f"Saved separated audio to {out_path}  shape={tuple(separated.shape)}")


if __name__ == "__main__":
    main()
