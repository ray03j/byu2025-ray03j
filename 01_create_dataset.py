import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    return args

def load_tomogram(tomo_dir, resize=(64, 64), fixed_depth=32):
    slices = []
    if os.path.isdir(tomo_dir):
        slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    else:
        slice_files = []  # ディレクトリが存在しない場合は空リストにしてスキップ

    if len(slice_files) < fixed_depth:
        padding_needed = fixed_depth - len(slice_files)
        empty_img = Image.new('L', resize, color=0)
        slices.extend([np.array(empty_img)] * padding_needed)
    else:
        slice_files = slice_files[:fixed_depth]  # truncate if more than fixed_depth

    for slice_file in slice_files:
        img = Image.open(os.path.join(tomo_dir, slice_file)).convert('L')
        img = img.resize(resize, Image.Resampling.LANCZOS)
        slices.append(np.array(img))

    return np.stack(slices).astype(np.float32) / 255.0

def main():
    args = get_args()
    mode = args.mode # train or test
    data_root = Path(__file__).resolve().parent
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = data_root.joinpath("data")
    df = pd.read_csv(input_dir.joinpath("train_labels.csv"))
    if args.output_dir:
        output_dir = Path(args.output_dir).joinpath(f"{mode}_imgs")
    else:
        output_dir = data_root.joinpath("working", f"{mode}_imgs")
    output_dir.mkdir(exist_ok=True, parents=True)

    for tomo_id, sub_df in tqdm(df.groupby("tomo_id")):
        tomo_dir = input_dir.joinpath(mode, tomo_id)
        tomogram = load_tomogram(tomo_dir)
        if tomogram is None:
            continue
        tomogram = tomogram.astype(np.float32)
        np.savez_compressed(output_dir.joinpath(f"{tomo_id}.npz"), tomogram=tomogram)

if __name__ == '__main__':
    main()
