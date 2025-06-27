import torch
import torch.nn as nn
import torch.nn.functional as F
from .timmEncoder import TimmEncoder

def get_model_from_cfg(cfg):
    if cfg.model.arch == "simple_2_5d_model":
        model = Simple2_5DModel(cfg)
    elif cfg.model.arch == "2.5d_unet":
        model = UNet2_5D(cfg)
    elif cfg.model.arch == "timm_encoder":
        model = TimmEncoder(cfg)
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model.arch}")
    return model


class Simple2_5DModel(nn.Module):
    def __init__(self, cfg):
        super(Simple2_5DModel, self).__init__()
        self.cfg = cfg
        
        # 2D畳み込み層 (画像の特徴を抽出)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16チャネル → 32チャネル
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32チャネル → 64チャネル
        self.pool = nn.MaxPool2d(2, 2)  # プーリング層で空間次元を削減
        
        # フラット化された特徴量を最終的な出力に結びつける
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # ここではプーリングを2回行っている前提
        self.fc2 = nn.Linear(128, 1)  # 出力層（クラス数に合わせる）

    def forward(self, x):
        x = x.squeeze(1)  # チャネル数が1の場合、次元を削除
        # 畳み込み層を適用
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        
        # フラット化
        x = x.view(x.size(0), -1)  # Flatten to 1D vector
        # 全結合層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        H, W = 32, 32
        # y_pred を [batch_size, 1, H, W] などの形に変換
        x = x.view(x.size(0), 1, H, W)  # H, W は必要に応じて調整
        return x


class UNet2_5D(nn.Module):
    def __init__(self, cfg):
        super(UNet2_5D, self).__init__()
        self.cfg = cfg
        
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.model.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, cfg.model.class_num, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x