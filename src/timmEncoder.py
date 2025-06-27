import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TimmEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        pretrained = True if cfg.model.resume_path is None else False

        self.encoder = timm.create_model(
            cfg.model.backbone,
            in_chans=cfg.task.slice_depth,
            pretrained=pretrained,
            drop_path_rate=cfg.model.drop_path_rate,
            features_only=True,
        )

        encoder_channels = self.encoder.feature_info.channels()
        self.out_feature_dim = encoder_channels[-1]

        self.regression_head = nn.Conv2d(self.out_feature_dim, 3, kernel_size=1)
        self.segmentation_head = nn.Conv2d(encoder_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        引数:
            x (Tensor): (バッチサイズ, fixed_depth, 高さ, 幅)
                        fixed_depthはスライス方向の枚数で、チャネルとして扱われる(例: 3層=3チャネル)
        戻り値:
            Dict[str, Tensor]: 以下を含む辞書:
                - seg (Tensor): セグメンテーション出力 (バッチサイズ, 1, 高さ, 幅)
                                各画素がモーターを含むかどうかのスコア（シグモイドを適用して使用）
                - reg (Tensor): 密な回帰出力 (バッチサイズ, 3, 高さ, 幅)
                                各画素ごとの (z, y, x) 方向の回帰ベクトル
        """
        # print(f"Input shape: {x.shape}, in_chans: {self.cfg.task.slice_depth}")
        bs, d, h, w = x.shape  # dはfixed_depth（チャネル数）

        # エンコーダー特徴量の取得
        features = self.encoder(x)
        last_feature = features[-1]  # (B, C, H', W')

        seg_output = self.segmentation_head(last_feature)
        reg_output = self.regression_head(last_feature)  # (B, 3, H', W')


        # セグメンテーション出力のサイズ調整
        if seg_output.shape[-2:] != (h, w):
            seg_output = F.interpolate(seg_output, size=(h, w), mode='bilinear', align_corners=False)
        if reg_output.shape[-2:] != (h, w):
            reg_output = F.interpolate(reg_output, size=(h, w), mode='bilinear', align_corners=False)

        return {
            'seg': seg_output,
            'reg': reg_output,
        }

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.encoder.model.set_grad_checkpointing(enable)
