from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import os
import warnings

def load_tomogram(tomo_dir, resize=(224, 224), fixed_depth=16):
    slices = []
    if not os.path.isdir(tomo_dir):
        slice_files = []  # スキップ
    else:
        slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    if len(slice_files) < fixed_depth:
        padding_needed = fixed_depth - len(slice_files)
        empty_img = Image.new('L', resize, color=0)
        slices.extend([np.array(empty_img)] * padding_needed)
    else:
        slice_files = slice_files[:fixed_depth]

    for slice_file in slice_files:
        img = Image.open(os.path.join(tomo_dir, slice_file)).convert('L')
        img = img.resize(resize, Image.Resampling.LANCZOS)
        slices.append(np.array(img))

    return np.stack(slices).astype(np.float32) / 255.0



class MyDataset(Dataset):
    def __init__(self, df, cfg, mode):
        assert mode in ["train", "val", "test"]
        self.cfg = cfg
        self.mode = mode
        self.tomogram_ids = df["tomo_id"].unique().tolist()
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)
        data_root = Path(__file__).parents[1].joinpath("data")
        self.input_dir = Path(cfg.data.input_dir) if hasattr(cfg.data, "input_dir") else Path(__file__).parents[1] / "data"
        self.output_dir = Path(cfg.data.output_dir) if hasattr(cfg.data, "output_dir") else Path(__file__).parents[1] / "working"
        self.data_dir = data_root / mode

        train_labels_path = self.input_dir / "train_labels.csv"
        self.train_labels_df = pd.read_csv(train_labels_path).drop_duplicates(subset='tomo_id', keep='first').set_index('tomo_id')

        if mode == "test":
            test_dir = self.input_dir / "test"
            self.tomogram_ids = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

        if mode == "train":
            self.tomogram_ids = list(self.train_labels_df.index)
        
        self.resize = (cfg.task.img_size, cfg.task.img_size)
        self.fixed_depth = cfg.task.fixed_depth
        self.slice_depth = cfg.task.slice_depth
        self.slice_indexes = []
        seen = set()

        for tomo_id in self.tomogram_ids:
            if tomo_id in seen:
                continue  # 重複するtomo_idはスキップ
            seen.add(tomo_id)

            npz_path = self.output_dir.joinpath("train_imgs", f"{tomo_id}.npz")
            if not npz_path.exists():
                print(f"[WARN] Missing npz for tomo_id={tomo_id}: {npz_path}")
                continue  # npzがない場合もスキップ（オプション）

            data = np.load(npz_path)
            tomogram = data["tomogram"]  # (D, H, W)
            depth = tomogram.shape[0]

            for i in range(depth - self.slice_depth + 1):
                self.slice_indexes.append((tomo_id, i))

    def __len__(self):
        return len(self.slice_indexes)
    
    def __getitem__(self, idx):
        tomo_id, start_idx = self.slice_indexes[idx]
        data_root = Path(__file__).parents[1]

        if self.mode == "test":
            npz_dir = self.output_dir.joinpath("test_imgs")
            npz_path = npz_dir.joinpath(f"{tomo_id}.npz")
            if not npz_path.exists():
                warnings.warn(f"Test NPZ file not found for tomo_id: {tomo_id}")
                # エラー処理またはスキップ
                # ここではダミーデータを返す例
                h, w = self.cfg.task.img_size, self.cfg.task.img_size
                sub_volume = torch.zeros((self.slice_depth, h, w), dtype=torch.float32)
                label = np.zeros(3, dtype=np.float32) # ダミーラベル
                mask = np.zeros((1, h, w), dtype=np.float32) # 全体が0のマスク
                offset = np.zeros((3, h, w), dtype=np.float32) # 全体が0のオフセット
                return (
                    sub_volume,
                    torch.from_numpy(label),
                    torch.from_numpy(mask),
                    torch.from_numpy(offset),
                    tomo_id,
                )

            data = np.load(npz_path)
            tomogram = data["tomogram"]
            sub_volume = tomogram[start_idx:start_idx + self.slice_depth]
            sub_volume = np.transpose(sub_volume, (1, 2, 0))
            if self.transforms is not None:
                sub_volume = self.transforms(image=sub_volume)["image"]

            h, w = self.cfg.task.img_size, self.cfg.task.img_size
            
            # ラベル・マスク・オフセット
            label = np.zeros(3, dtype=np.float32) 
            mask = np.ones((1, h, w), dtype=np.float32) 
            offset = np.zeros((3, h, w), dtype=np.float32) 

            return (
                sub_volume,
                torch.from_numpy(label.astype(np.float32)),
                torch.from_numpy(mask),
                torch.from_numpy(offset),
                tomo_id,
            )
        
        if self.mode == "train" or self.mode == "val":
            # train/valデータの場合は入力画像とラベルを返す
            npz_dir = self.output_dir.joinpath("train_imgs")
            npz_path = npz_dir.joinpath(f"{tomo_id}.npz")
            data = np.load(npz_path)
            tomogram = data["tomogram"]

            # slice_depth分の層を取り出す
            sub_volume = tomogram[start_idx:start_idx + self.slice_depth]  # shape: (3, H, W)

            # [3, H, W] → [H, W, 3] にして albumentations に渡す
            sub_volume = np.transpose(sub_volume, (1, 2, 0))  # CHW → HWC
            if self.transforms is not None:
                sub_volume = self.transforms(image=sub_volume)["image"]  # torch.Tensor (3, H, W)


            # ラベルの取得
            label_row = self.train_labels_df.loc[tomo_id]
            label = label_row[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values
            
            # マスクとオフセットの生成
            h, w = self.cfg.task.img_size, self.cfg.task.img_size
            mask = self.create_mask_from_label(label_row, (h, w))
            offset = self.create_offset_from_label(label_row, (h, w))
            
            # self.transforms の適用
            if self.transforms is not None:
                transformed = self.transforms(image=tomogram)
                tomogram = transformed["image"]
            
            return (
                sub_volume, 
                torch.from_numpy(label.astype(np.float32)), 
                torch.from_numpy(mask), 
                torch.from_numpy(offset),
            )


    def create_mask_from_label(self, label_data, size):
        """ラベルデータからマスクを生成"""
        mask = np.zeros((1, size[0], size[1]), dtype=np.float32)
        
        try:
            # モーター軸の位置を画像サイズに合わせて正規化
            motor_y = int(label_data['Motor axis 0'] * size[0] / label_data['Array shape (axis 0)'])
            motor_x = int(label_data['Motor axis 1'] * size[1] / label_data['Array shape (axis 1)'])
            
            # モーター位置の周囲にマスクを生成
            radius = 5  # NOTE:マスクの半径
            y_min = max(0, motor_y - radius)
            y_max = min(size[0], motor_y + radius + 1)
            x_min = max(0, motor_x - radius)
            x_max = min(size[1], motor_x + radius + 1)
            
            mask[0, y_min:y_max, x_min:x_max] = 1.0
            
        except Exception as e:
            print(f"マスク生成エラー: {e}")
        
        return mask

    def create_offset_from_label(self, label_data, size):
        """ラベルデータから3チャネルのオフセットマップを生成"""
        offset = np.zeros((3, size[0], size[1]), dtype=np.float32)
        
        try:
            # モーター位置を正規化座標に変換
            norm_z = label_data['Motor axis 0'] / label_data['Array shape (axis 0)']
            norm_y = label_data['Motor axis 1'] / label_data['Array shape (axis 1)']
            norm_x = label_data['Motor axis 2'] / label_data['Array shape (axis 2)']
            
            # 座標グリッドを生成
            y_coords, x_coords = np.mgrid[0:size[0], 0:size[1]]
            y_coords = y_coords.astype(np.float32) / size[0]
            x_coords = x_coords.astype(np.float32) / size[1]
            
            # Zオフセットは全ピクセル同じ値（norm_z）にする例
            offset[0] = norm_z  # z方向オフセット (固定値)
            
            # 各ピクセルからモーター位置へのオフセットを計算
            offset[1] = norm_x - x_coords  # X方向のオフセット
            offset[2] = norm_y - y_coords  # Y方向のオフセット
            
        except Exception as e:
            print(f"オフセット生成エラー: {e}")
        
        return offset


def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.XYMasking(num_masks_x=(1, 4), num_masks_y=(1, 4), mask_y_length=(0, 32), mask_x_length=(0, 32),
            #             fill_value=-1.0, p=0.5),
        
            # A.CenterCrop(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.05, scale_limit=0.1, value=0,
            #                    rotate_limit=180, mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.model.img_size, min_width=cfg.model.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            # A.RandomRotate90(p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # normalize with imagenet statis
            # A.Normalize(p=1.0, mean=5.2577832e-08, std=7.199929e-06, max_pixel_value=1.0),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.3, contrast_limit=0.3, p=0.3
            # ),
            ToTensorV2(p=1.0, transpose_mask=True),
        ],
        p=1.0,
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.RandomScale(scale_limit=(1.0, 1.0), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Crop(y_max=self.cfg.data.val_img_h, x_max=self.cfg.data.val_img_w, p=1.0),
            # A.Normalize(p=1.0, mean=23165, std=2747),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
