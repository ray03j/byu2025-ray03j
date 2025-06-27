from pathlib import Path
import numpy as np
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch

from .model import get_model_from_cfg
from .loss import get_loss
from .util import mixup, get_augment_policy


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.preds = None
        self.gts = None
        self.cfg = cfg
        self.mode = mode
        self.model = get_model_from_cfg(cfg)
        self.model_ema = None

        self.input_dir = Path(cfg.data.input_dir) if hasattr(cfg.data, "input_dir") else Path(__file__).parents[1] / "data"
        self.output_dir = Path(cfg.data.output_dir) if hasattr(cfg.data, "output_dir") else Path(__file__).parents[1] / "working"

        self.loss = get_loss(cfg)
        self.validation_step_outputs = []


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, label, mask, offset = batch  # x: (D, H, W), label: (2, H, W), mask: (1, H, W), offset: ( , , )

        output = self(x) # {'seg': (B, 1, H, W), 'reg': (B, 3, H, W)} の辞書を返す
        loss_dict = self.loss(output, label, mask, offset)

        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        x, label, mask, offset = batch  # x: (D, H, W), label: (2, H, W), mask: (1, H, W), offset: ( , , )

        output = self(x)
        loss_dict = self.loss(output, label, mask, offset)

        for k, v in loss_dict.items():
            self.log(f"val_{k}", v, on_step=True, on_epoch=True, sync_dist=True)
        
        self.validation_step_outputs.append(loss_dict)
        return loss_dict

    def on_validation_epoch_start(self) -> None:
        pass
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
    
        if outputs:
            # エポック全体の平均損失を計算
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            # 明示的にval_lossとしてログ
            self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.test_outputs = []
        self.test_targets = []
        self.test_ids = []

    def test_step(self, batch, batch_idx):
        x, label, mask, offset, tomo_id = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        self.test_outputs.append({
            'seg': output['seg'].detach().cpu().numpy(),  # (B, 3, H, W)
            'reg': output['reg'].detach().cpu().numpy(),  # (B, 3)
        })
        
        if mask is not None and label is not None:
            self.test_targets.append({
                'mask': mask.detach().cpu().numpy(),
                'label': label.detach().cpu().numpy(),
                'offset': offset.detach().cpu().numpy()
            })
        
        self.test_ids.append(batch_idx)

        loss_dict = self.loss(output, label, mask, offset)

        log_dict = {f"test_{k}": v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_test_epoch_end(self):
        # テスト結果の集約
        all_outputs = {
            'seg': np.concatenate([out['seg'] for out in self.test_outputs], axis=0),
            'reg': np.concatenate([out['reg'] for out in self.test_outputs], axis=0),
        }

        if self.test_targets:
            all_targets = {
                'mask': np.concatenate([t['mask'] for t in self.test_targets], axis=0),
                'offset': np.concatenate([t['offset'] for t in self.test_targets], axis=0) if isinstance(self.test_targets[0]['offset'], np.ndarray) else None
            }
        else:
            all_targets = None

        # 結果の保存
        filename = Path(self.cfg.model.resume_path).stem
        output_path = self.output_dir.joinpath(f"test_results__{filename}.npz")
        
        save_dict = {'outputs': all_outputs}
        if all_targets is not None:
            save_dict['targets'] = all_targets
        save_dict['ids'] = np.array(self.test_ids)
        
        np.savez(output_path, **save_dict)

    def predict_step(self, batch):
        x, label, mask, offset, tomo_id = batch
        
        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        seg_output = output['seg']  # (B, 1, H, W)
        reg_output = output['reg']  # (B, 3, H, W)
        
        batch_size = seg_output.shape[0]
        tomo_ids = []
        coords_list = []
        
        for i in range(batch_size):
            # 各バッチ要素に対して予測を生成
            seg = seg_output[i]  # (out_channels, H, W)
            reg = reg_output[i]  # (3, H, W)
            
            # セグメンテーション出力の確信度を計算
            confidence = torch.sigmoid(seg)  # (out_channels, H, W)
            
            # 最も確信度の高い位置を見つける
            max_conf_val, _ = confidence.max(dim=0)  # (H, W)
            max_conf_pos = torch.where(max_conf_val == max_conf_val.max())
            y, x = [pos[0].item() for pos in max_conf_pos]
            
            # その位置での回帰出力を取得（z, x, y座標）
            coords = reg[:, y, x]  # (3,) - z, x, y coordinates
            
            # 座標を0-1の範囲に正規化
            normalized_coords = torch.stack([
                coords[0],  # z座標（回帰出力から直接取得）
                coords[1] / reg.shape[1],  # x座標を正規化
                coords[2] / reg.shape[2]   # y座標を正規化
            ])
            
            tomo_ids.append(tomo_id[i])
            coords_list.append(normalized_coords)

        # バッチ内の全予測を結合
        coords_tensor = torch.stack(coords_list)
        return list(zip(tomo_ids, coords_tensor.detach().cpu().numpy()))

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
                                                    warmup_lr=self.cfg.opt.lr / 10.0, **self.cfg.scheduler)
        
        lr_dict = dict(
            scheduler=scheduler,
            interval="epoch",  # same as default
            frequency=1,  # same as default
        )

        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
        # scheduler.step_update(num_updates=self.global_step)
