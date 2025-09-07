import argparse
import logging
from functools import reduce
from pathlib import Path

import torch
import wandb
import yaml
from kornia.metrics import AverageMeter
from torch import Tensor
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR

import loader
from config import from_dict, ConfigDict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device


class TrainF:
    def __init__(self, config: str | Path | ConfigDict, wandb_key: str):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Training Script')

        # init config
        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r'))
            config = from_dict(config)  # convert dict to object
        else:
            config = config
        self.config = config

        # debug mode
        if config.debug.fast_run:
            logging.warning('fast run mode is on, only for debug!')

        # wandb run
        wandb.login(key=wandb_key)  # wandb api key
        runs = wandb.init(project='TarDAL-v1', config=config, mode=config.debug.wandb_mode)
        self.runs = runs

        # init save folder
        save_dir = Path(self.config.save_dir) / self.runs.id
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        logging.info(f'model weights will be saved to {str(save_dir)}')

        # init pipeline
        fuse = Fuse(config, mode='train')
        self.fuse = fuse

        # freeze & grad
        for k, v in fuse.generator.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in config.train.freeze):
                logging.info(f'freezing {k}')
                v.requires_grad = False

        # init optimizer
        o_cfg = config.optimizer

        # 1. 生成器参数（所有需要训练的参数）
        generator_params = [p for p in self.fuse.generator.parameters() if p.requires_grad]

        # 2. 判别器参数（目标判别器+细节判别器）
        discriminator_params = []
        discriminator_params.extend([p for p in self.fuse.dis_t.parameters() if p.requires_grad])
        discriminator_params.extend([p for p in self.fuse.dis_d.parameters() if p.requires_grad])

        # 确保生成器和判别器参数无重叠
        gen_ids = {id(p) for p in generator_params}
        dis_ids = {id(p) for p in discriminator_params}
        assert gen_ids.isdisjoint(dis_ids), "生成器和判别器参数存在重叠！"

        # 定义参数组（仅分两组：生成器+判别器）
        groups = [
            # 生成器参数（基础学习率）
            {'params': generator_params, 'lr': o_cfg.lr_i, 'weight_decay': o_cfg.weight_decay},
            # 判别器参数（学习率为生成器的2倍，加速对抗训练）
            {'params': discriminator_params, 'lr': o_cfg.lr_i * 2, 'weight_decay': o_cfg.weight_decay / 10}
        ]

        # 初始化优化器
        # 初始化优化器（修复参数组重复）
        match o_cfg.name:
            case 'sgd':
                # 直接将groups传入优化器，无需后续add_param_group
                self.optimizer = SGD(groups, momentum=o_cfg.momentum, nesterov=True)
            case 'adam':
                self.optimizer = Adam(groups, betas=(o_cfg.momentum, 0.999))
            case 'adamw':
                self.optimizer = AdamW(groups, betas=(o_cfg.momentum, 0.999))
            case _:
                assert NotImplemented, f'不支持的优化器: {o_cfg.name}'

        # self.optimizer.add_param_group(groups[0])
        # self.optimizer.add_param_group(groups[1])

        # init scheduler
        # lr_fn = lambda x: (1 - x / config.train.epochs) * (1 - o_cfg.lr_f) + o_cfg.lr_f
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        # === 修改：初始化 ReduceLROnPlateau 调度器 ===
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 监控指标越小越好
            factor=0.5,  # 学习率衰减因子
            patience=5,  # 多少个epoch无改善后衰减
            min_lr=o_cfg.lr_f,  # 最小学习率
            threshold=0.0001,  # 改善阈值
            threshold_mode='rel',  # 相对改善
            cooldown=3,  # 衰减后的冷却期
        )

        # init dataset & dataloader
        data_t = getattr(loader, config.dataset.name)  # dataset type
        t_dataset = data_t(root=config.dataset.root, mode='train', config=config)
        v_dataset = data_t(root=config.dataset.root, mode='val', config=config)
        self.t_loader = DataLoader(
            t_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )
        self.v_loader = DataLoader(
            v_dataset, batch_size=config.train.batch_size,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )

    def run(self):
        # epochs & eval interval & save interval
        epochs = self.config.train.epochs
        e_interval = self.config.train.eval_interval
        s_interval = self.config.train.save_interval

        # 新增：预训练配置，添加分阶段训练（冻结判别器）
        pretrain_epochs = 100  # 前100轮冻结判别器
        current_adv_weight = 0.0  # 对抗损失权重

        # start training process
        for epoch in range(1, epochs + 1):
            # === 新增：初始化损失累加器 ===
            total_g_loss = 0.0
            step_count = 0
            # 新增：判别器冻结/解冻逻辑
            if epoch <= pretrain_epochs:
                # 预训练阶段：冻结判别器
                self.fuse.dis_t.eval()
                self.fuse.dis_d.eval()
                for param in self.fuse.dis_t.parameters():
                    param.requires_grad = False
                for param in self.fuse.dis_d.parameters():
                    param.requires_grad = False
                current_adv_weight = 0.0  # 禁用对抗损失
            else:
                # 联合训练阶段：解冻判别器
                self.fuse.dis_t.train()
                self.fuse.dis_d.train()
                for param in self.fuse.dis_t.parameters():
                    param.requires_grad = True
                for param in self.fuse.dis_d.parameters():
                    param.requires_grad = True
                current_adv_weight = self.config.loss.fuse.adv  # 使用配置权重

            # train
            t_l = tqdm(self.t_loader, disable=False, total=len(self.t_loader) if not self.config.debug.fast_run else 3, ncols=120)
            g_history = [AverageMeter() for _ in range(5)]  # tot, src, adv, tar, det
            disc_history = AverageMeter(), AverageMeter()  # target, detail
            log_dict = {}
            # 新增：梯度监控
            grad_norms = []
            for sample in t_l:
                sample = dict_to_device(sample, self.fuse.device)
                # 额外验证
                assert sample['ir'].device.type == 'cuda', "红外图像未移至CUDA"
                assert sample['vi'].device.type == 'cuda', "可见光图像未移至CUDA"
                # train generator
                # g_loss, [src_l, adv_l, tar_l, det_l] = self.fuse.criterion_generator(
                #     ir=sample['ir'], vi=sample['vi'],
                #     mk=sample['mask'],
                #     w1=sample['ir_w'], w2=sample['vi_w'],
                #     d_warming=epoch <= self.config.loss.fuse.d_warm,
                # )
                # 修改生成器损失调用（传递对抗权重）
                g_loss, [src_l, adv_l, tar_l, det_l] = self.fuse.criterion_generator(
                    ir=sample['ir'], vi=sample['vi'], mk=sample['mask'],
                    w1=sample['ir_w'], w2=sample['vi_w'],
                    d_warming=epoch <= self.config.loss.fuse.d_warm,
                    current_adv_weight=current_adv_weight  # 新增参数
                )

                g_history[0].update(g_loss.item())
                _ = [g_history[idx + 1].update(v) for idx, v in enumerate([src_l, adv_l, tar_l, det_l])]
                self.optim(g_loss)

                # === 新增：累加生成器损失 ===
                total_g_loss += g_loss.item()
                step_count += 1

                # train target discriminator
                d_t_loss = self.fuse.criterion_dis_t(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[0].update(d_t_loss.item())
                # self.optim(d_t_loss)
                self.optim(d_t_loss, discriminator=True)  # 添加判别器标识
                # train detail discriminator
                d_d_loss = self.fuse.criterion_dis_d(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[1].update(d_d_loss.item())
                # self.optim(d_d_loss)
                self.optim(d_d_loss, discriminator=True)  # 添加判别器标识


                # fast run (jump out)
                if self.config.debug.fast_run and t_l.n > 2:
                    logging.info('fast mode: jump')
                    break
                # 梯度监控
                total_norm = 0
                for p in self.fuse.generator.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)
                self.runs.log({"grad_norm": grad_norms[-1]})
            # train logs
            g_l, src_l, adv_l, tar_l, det_l = [g_history[i].avg for i in range(5)]
            d_t_l, d_d_l = disc_history[0].avg, disc_history[1].avg
            log_dict |= {'g/tot': g_l, 'g/src': src_l, 'g/adv': adv_l, 'g/tar': d_t_l, 'g/det': d_d_l, 'disc/tar': tar_l, 'disc/det': det_l}
            logging.info(f'Epoch {epoch}/{epochs} | Generator Loss: {g_l:.4f} | Source Loss: {src_l:.4f} | Adversarial Loss: {adv_l:.4f}')

            # 图片日志由 save_interval (s_interval) 控制，与模型保存同步
            if epoch % s_interval == 0 or self.config.debug.fast_run:  # 关键修改：e_interval → s_interval
                e_l = tqdm(self.v_loader, disable=True)
                for sample in e_l:
                    sample = dict_to_device(sample, self.fuse.device)
                    fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
                    # 建议给图片日志添加明确名称，避免与其他日志混淆
                    log_dict |= {
                        'fuse/saved': wandb.Image(fus, caption=f'Epoch {epoch}'),
                        'mask/saved': wandb.Image(sample['mask'], caption=f'Epoch {epoch}')
                    }
                    break
            # update scheduler and show lr
            log_dict |= reduce(lambda x, y: x | y, [{f'lr_{i}': v['lr']} for i, v in enumerate(self.optimizer.param_groups)])

            # === 新增：计算平均生成器损失 ===
            avg_g_loss = total_g_loss / step_count if step_count > 0 else 0

            # === 修改：更新学习率调度器 ===
            # 使用平均生成器损失作为指标
            self.scheduler.step(avg_g_loss)

            # update wandb
            self.runs.log(log_dict)
            # save model
            if epoch % s_interval == 0 or self.config.debug.fast_run:
                ckpt = self.fuse.save_ckpt()
                torch.save(ckpt, self.save_dir / f'{str(epoch).zfill(5)}.pth')
                logging.info(f'Epoch {epoch}/{epochs} | Model Saved')

    # def optim(self, loss: Tensor):
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def optim(self, loss: Tensor, discriminator: bool = False):
        """
        添加梯度裁剪
        """
        self.optimizer.zero_grad()
        loss.backward()

        # 动态梯度裁剪阈值（核心改进）
        max_norm = max(1.0, 0.01 * abs(loss.item()))  # 随损失动态调整

        if discriminator:
            params = list(self.fuse.dis_t.parameters()) + list(self.fuse.dis_d.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.fuse.generator.parameters(), max_norm=max_norm)

        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--auth', help='wandb auth api key')
    args = parser.parse_args()
    train = TrainF(args.cfg, args.auth)
    train.run()
