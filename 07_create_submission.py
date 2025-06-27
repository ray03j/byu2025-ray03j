import argparse
from omegaconf import OmegaConf

from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
import csv

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    if args.output_dir is not None:
        cfg.data.output_dir = args.output_dir
    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力ファイルの取得
    input_file = Path(output_dir).joinpath("output.csv")
    output_file = output_dir / "submission.csv"

    tomo_coords = defaultdict(list)

    # 読み込み
    with open(input_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # ヘッダーをスキップ
        for row in reader:
            tomo_id = row[0]
            coords = list(map(float, row[1:]))
            tomo_coords[tomo_id].append(coords)

    # 書き出し
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"])
        for tomo_id, coords_list in tomo_coords.items():
            mean_coords = np.mean(coords_list, axis=0)
            writer.writerow([tomo_id] + mean_coords.tolist())

if __name__ == '__main__':
    main()
