#!/usr/bin/env python
"""
Download FACodec checkpoint from HuggingFace.
Run with: pixi run python scripts/download_facodec.py
"""
import os
import sys
from pathlib import Path

def main():
    # Add FAcodec to path
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "FAcodec"))

    from hf_utils import load_custom_model_from_hf

    print("Downloading FACodec checkpoint from HuggingFace...")
    print("This may take a few minutes (checkpoint is ~1.4GB)")

    # Download to ./checkpoints (relative to repo root)
    os.chdir(repo_root)
    model_path, config_path = load_custom_model_from_hf('Plachta/FAcodec')

    print(f"\nDownloaded files:")
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")

    # Copy to FAcodec directory
    import shutil

    facodec_ckpt_dir = repo_root / "FAcodec" / "checkpoints"
    facodec_cfg_dir = repo_root / "FAcodec" / "configs"

    facodec_ckpt_dir.mkdir(exist_ok=True)
    facodec_cfg_dir.mkdir(exist_ok=True)

    # Copy files (follow symlinks)
    target_ckpt = facodec_ckpt_dir / "FAcodec.pth"
    target_cfg = facodec_cfg_dir / "config.yml"

    shutil.copy2(model_path, target_ckpt)
    shutil.copy2(config_path, target_cfg)

    print(f"\nCopied to expected locations:")
    print(f"  Checkpoint: {target_ckpt}")
    print(f"  Config: {target_cfg}")
    print("\nDone!")

if __name__ == "__main__":
    main()
