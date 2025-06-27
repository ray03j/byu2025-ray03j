from pathlib import Path
import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split

from .dataset import MyDataset
import os 


class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.val_ratio = 0.3
        self.test_dataset = None
        self.input_dir = Path(cfg.data.input_dir) if hasattr(cfg.data, "input_dir") else Path(__file__).parents[1] / "data"
        self.output_dir = Path(cfg.data.output_dir) if hasattr(cfg.data, "output_dir") else Path(__file__).parents[1] / "working"
        self.val_output_dir = (
            Path(cfg.data.val_output_dir)
            if hasattr(cfg.data, "val_output_dir") and cfg.data.val_output_dir is not None
            else self.output_dir / "val"
        )
        
        self.train_ids = None
        self.val_ids = None

    def prepare_data(self):
        """データセットの準備を行う。train から val を作成する"""
        # train_labels.csv からデータを読み込む
        df = pd.read_csv(self.input_dir.joinpath("train_labels.csv"))
        tomo_ids = df["tomo_id"].unique()
        
        # train/val の分割 - NOTE: trainはすでにtrainディレクトリにある
        self.train_ids, self.val_ids = train_test_split(tomo_ids, test_size=self.val_ratio, random_state=42)
        
        # valディレクトリの作成
        val_dir = self.val_output_dir
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # trainディレクトリから valディレクトリにデータをコピー
        train_dir = self.input_dir.joinpath("train")
        for tomo_id in self.val_ids:
            src_dir = train_dir.joinpath(tomo_id)
            dst_dir = val_dir.joinpath(tomo_id)
            
            if src_dir.exists() and not (dst_dir.exists() or dst_dir.is_symlink()):
                os.symlink(src_dir, dst_dir)
                print(f"Created symlink for {tomo_id}: {src_dir} -> {dst_dir}")
            else:
                print(f"Skipping {tomo_id}, symlink or directory already exists.")

    def setup(self, stage=None):

        if stage == "fit":
            df = pd.read_csv(self.input_dir.joinpath("train_labels.csv"))
            npz_dir = self.output_dir.joinpath("train_imgs")

            if self.train_ids is None or self.val_ids is None:
                warnings.warn("prepare_data() が先に呼ばれていません。setup() はスキップされます。")
                self.train_dataset = None
                self.val_dataset = None
                return
            
            # npzファイルの存在確認処理
            required_tomo_ids = set(self.train_ids) | set(self.val_ids)
            missing_files = []            
            for tomo_id in required_tomo_ids:
                npz_path = npz_dir.joinpath(f"{tomo_id}.npz")
                if not npz_path.is_file():
                    missing_files.append(tomo_id)
            if missing_files:
                warnings.warn(f"必要なNPZファイルが見つかりません: {missing_files}")
                self.train_dataset = None
                self.val_dataset = None
                return

            self.train_dataset = MyDataset(df[df["tomo_id"].isin(self.train_ids)], self.cfg, mode="train")
            self.val_dataset = MyDataset(df[df["tomo_id"].isin(self.val_ids)], self.cfg, mode="val")

        if stage == "test" or stage == "predict":
            test_dir = self.input_dir.joinpath("test")
            test_ids = [f.name for f in test_dir.iterdir() if f.is_dir()]
            if not test_ids:
                warnings.warn("テストディレクトリが空です。")
                self.test_dataset = None
                return

            npz_dir = self.output_dir.joinpath("test_imgs")
            missing_files = []
            for tomo_id in test_ids:
                npz_path = npz_dir.joinpath(f"{tomo_id}.npz")
                if not npz_path.is_file():
                    missing_files.append(tomo_id)
            if missing_files:
                warnings.warn(f"以下のtomo_idのNPZファイルが見つかりません: {missing_files}")
                self.test_dataset = None
                return
            
            data = [{"tomo_id": tomo_id} for tomo_id in test_ids]
            test_df = pd.DataFrame(data)
            self.test_dataset = MyDataset(test_df, self.cfg, mode="test")
        
    def train_dataloader(self):    
        return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
