import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import os

from src.datamodule import MyDataModule
from src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_output_dir", type=str, default=None, help="Validation データの出力先")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory path. If not specified, tries to find in /kaggle/input.") # この行はそのまま
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def find_input_dir():
    base_dir = Path("/kaggle/input")
    if not base_dir.exists():
        return None
    
    for subdir in base_dir.iterdir():
        if (subdir / "test").exists():
            return str(subdir)
    return None 


def process_predictions(predictions):
    tomo_ids = []
    coords = []

    for batch_preds in predictions:
        for tomo_id, pred in batch_preds:
            tomo_ids.append(tomo_id)
            coords.append([pred[0], pred[2], pred[1]])  # z, y, x

    if not coords:
        raise ValueError("No predictions were made.")
    coords = np.array(coords)
    df = pd.DataFrame({
        "tomo_id": tomo_ids,
        "Motor axis 0": coords[:, 0],  # z
        "Motor axis 1": coords[:, 1],  # y
        "Motor axis 2": coords[:, 2],  # x
    })
    return df


def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    if args.input_dir is not None:
        cfg.data.input_dir = args.input_dir
    else:
        found_path = find_input_dir()
        if found_path:
            cfg.data.input_dir = found_path
        else:
            raise FileNotFoundError(
                "Input directory not specified via --input_dir, "
                "and could not be found in /kaggle/input. "
                "Please specify input directory using --input_dir argument."
            )
    
    if args.output_dir is not None:
        cfg.data.output_dir = args.output_dir
    if args.val_output_dir is not None:
        cfg.data.val_output_dir = args.val_output_dir
    
    if not hasattr(cfg.data, "output_dir") or cfg.data.output_dir is None:
        cfg.data.output_dir = Path(__file__).parent.joinpath("working").as_posix() # デフォルトパスを設定

    cfg.model.resume_path = args.checkpoint

    model = MyModel(cfg, mode="predict") # ここで cfg.data.input_dir が参照される
    dm = MyDataModule(cfg)

    trainer = Trainer(**cfg.trainer)
    predictions = trainer.predict(model, datamodule=dm)
    
    # test/ディレクトリが存在するか確認
    test_dir = Path(cfg.data.input_dir).joinpath("test")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}. Please check your input_dir setting.")

    # Submission作成と保存
    submission = process_predictions(predictions)
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "output.csv", index=False)
    print(f"Submission saved to {output_dir}/output.csv")


if __name__ == "__main__":
    main()
