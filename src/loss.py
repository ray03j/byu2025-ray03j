import torch
from torch import nn
import torch.nn.functional as F


def get_loss(cfg):
    return MyLoss(cfg)

class MyLoss(nn.Module):
    def __init__(self, cfg):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.criterion_seg = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.MSELoss(reduction="mean")
        self.lambda_coord = 1.0

    def forward(self, y_pred, t_pred, mask, offset):
        """
        Args:
            y_pred (dict): {'seg': Tensor, 'reg': Tensor}
            t_pred (Tensor[float32]): (batch_size, 2, h, w)
            mask   (Tensor[int]):     (batch_size, 1, h, w)
            offset (Tensor[float32]): (batch_size, 2, h, w)
        Returns:
            dict: 'loss', 'loss_seg', 'loss_reg' を含む辞書
        """
        return_dict = {}

        seg_pred = y_pred['seg']
        reg_pred = y_pred['reg']

        if self.cfg.task.pretrain:
            loss = self.criterion_reg(seg_pred, mask.to(torch.float32))
            return_dict["loss"] = loss
            return return_dict

        # 形状の検証
        assert seg_pred.shape == mask.shape, f"seg_pred: {seg_pred.shape}, mask: {mask.shape}"
        assert reg_pred.shape == offset.shape, f"reg_pred: {reg_pred.shape}, offset: {offset.shape}"

        # 損失の計算
        y = mask.to(torch.float32)
        loss_seg = self.criterion_seg(seg_pred, y)

        # マスクをreg_predと同じチャネル数に拡張
        idx = mask.to(torch.bool).expand(-1, reg_pred.shape[1], -1, -1)
        
        # マスクが全てゼロの場合の処理
        if idx.any():
            loss_reg = self.criterion_reg(reg_pred[idx], offset[idx])
        else:
            # マスクが全てゼロの場合は回帰損失を0とする
            loss_reg = torch.tensor(0.0, device=seg_pred.device)

        loss_total = loss_seg + self.lambda_coord * loss_reg

        return {
            "loss_seg": loss_seg,
            "loss_reg": loss_reg,
            "loss": loss_total,
        }


def main():
    pass


if __name__ == '__main__':
    main()
