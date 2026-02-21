#!/usr/bin/env python
"""
FSDP 多模态对比预训练脚本

实现简化版 CLIP 模型的分布式训练，展示 FSDP (Fully Sharded Data Parallel) 的核心用法。

使用方法:
    # 单机 2 卡
    torchrun --nproc_per_node=2 experiments/scripts/fsdp_pretrain.py

    # 单机 4 卡，自定义配置
    torchrun --nproc_per_node=4 experiments/scripts/fsdp_pretrain.py \
        --epochs 30 --batch_size 128 --strategy full_shard

    # 查看所有参数
    python experiments/scripts/fsdp_pretrain.py --help

环境要求:
    - PyTorch >= 2.0
    - NCCL 后端（Linux + NVIDIA GPU）
    - torchvision
"""

import argparse
import functools
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


# ============================================================
# 数据定义
# ============================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

CAPTION_TEMPLATES = [
    'a photo of a {}',
    'a picture of a {}',
    'an image showing a {}',
    'a {} in a photograph',
    'a blurry photo of a {}',
    'a close-up of a {}',
    'a bright photo of a {}',
    'a dark photo of a {}',
]


class SimpleTokenizer:
    """简易词级别分词器"""

    def __init__(self, max_len=12):
        self.max_len = max_len
        words = set()
        for cls_name in CIFAR10_CLASSES:
            for template in CAPTION_TEMPLATES:
                for word in template.format(cls_name).split():
                    words.add(word)
        self.word2idx = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        for i, word in enumerate(sorted(words), start=3):
            self.word2idx[word] = i
        self.vocab_size = len(self.word2idx)

    def encode(self, text):
        tokens = [self.word2idx['<bos>']]
        tokens += [self.word2idx.get(w, 0) for w in text.split()]
        tokens += [self.word2idx['<eos>']]
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return tokens


class CIFAR10WithCaptions(Dataset):
    """带文本描述的 CIFAR-10 数据集"""

    def __init__(self, root='./data', train=True, tokenizer=None):
        self.dataset = datasets.CIFAR10(
            root=root, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]),
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        template = random.choice(CAPTION_TEMPLATES)
        caption = template.format(CIFAR10_CLASSES[label])
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        return image, tokens, label


# ============================================================
# 模型定义（与 notebook 中一致）
# ============================================================

class PatchEmbedding(nn.Module):
    """将图像分割为 patch 并映射到嵌入空间"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """标准 Pre-Norm Transformer 编码器层

    这个类是 FSDP auto_wrap_policy 的包装单元。
    FSDP 会以 TransformerBlock 为粒度进行参数分片。
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        normed = self.norm1(x)
        x = x + self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """简化版 ViT"""

    def __init__(self, img_size=32, patch_size=4, embed_dim=256,
                 num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


class TextEncoder(nn.Module):
    """简化版 Text Transformer"""

    def __init__(self, vocab_size, max_len=12, embed_dim=256,
                 num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        padding_mask = (x == 0)
        x = self.token_embed(x) + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        mask = (~padding_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x


class MiniCLIP(nn.Module):
    """简化版 CLIP 模型"""

    def __init__(self, vocab_size, img_size=32, patch_size=4,
                 embed_dim=256, proj_dim=128,
                 v_layers=6, t_layers=4, num_heads=8):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            img_size, patch_size, embed_dim, v_layers, num_heads,
        )
        self.text_encoder = TextEncoder(
            vocab_size, max_len=12, embed_dim=embed_dim,
            num_layers=t_layers, num_heads=num_heads,
        )
        self.vision_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, images, tokens):
        img_features = self.vision_encoder(images)
        txt_features = self.text_encoder(tokens)
        img_embed = F.normalize(self.vision_proj(img_features), dim=-1)
        txt_embed = F.normalize(self.text_proj(txt_features), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * img_embed @ txt_embed.t()
        return logits, img_embed, txt_embed


def contrastive_loss(logits):
    """对称 InfoNCE 损失"""
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


# ============================================================
# FSDP 配置
# ============================================================

STRATEGY_MAP = {
    'full_shard': ShardingStrategy.FULL_SHARD,       # ZeRO-3
    'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP, # ZeRO-2
    'no_shard': ShardingStrategy.NO_SHARD,           # DDP
}


def setup_fsdp_model(model, local_rank, args):
    """用 FSDP 包装模型

    Parameters
    ----------
    model : nn.Module
        原始模型（未包装）
    local_rank : int
        当前进程的本地 GPU 编号
    args : argparse.Namespace
        命令行参数，包含 strategy 和 use_mp

    Returns
    -------
    FSDP 包装后的模型
    """
    # 1. 自动包装策略: 以 TransformerBlock 为单位分片
    # 每个 TransformerBlock 内部的参数作为一个 FSDP 单元
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # 2. 混合精度配置（可选）
    mp_policy = None
    if args.use_mp:
        # 检测 BF16 支持（Ampere+ GPU: A100, H100 等）
        if torch.cuda.is_bf16_supported():
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            if local_rank == 0:
                print('[FSDP] 使用 BF16 混合精度')
        else:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if local_rank == 0:
                print('[FSDP] 使用 FP16 混合精度 (GPU 不支持 BF16)')

    # 3. 分片策略
    sharding = STRATEGY_MAP.get(args.strategy, ShardingStrategy.FULL_SHARD)
    if local_rank == 0:
        print(f'[FSDP] 分片策略: {sharding}')

    # 4. 包装模型
    model = FSDP(
        model,
        sharding_strategy=sharding,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
    )

    return model


# ============================================================
# 训练与保存
# ============================================================

def train_one_epoch(model, dataloader, optimizer, epoch, local_rank):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, (images, tokens, _) in enumerate(dataloader):
        images = images.to(local_rank)
        tokens = tokens.to(local_rank)

        logits, _, _ = model(images, tokens)
        loss = contrastive_loss(logits)

        optimizer.zero_grad()
        loss.backward()

        # FSDP 下的梯度裁剪: 必须用 model.clip_grad_norm_
        # 而非 torch.nn.utils.clip_grad_norm_
        model.clip_grad_norm_(max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if local_rank == 0 and batch_idx % 50 == 0:
            print(f'  [{batch_idx:>3d}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f}')

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches

    if local_rank == 0:
        print(f'Epoch {epoch} | 平均 Loss: {avg_loss:.4f} | '
              f'耗时: {elapsed:.1f}s | '
              f'吞吐: {len(dataloader.dataset) / elapsed:.0f} 样本/秒')

    return avg_loss


def save_checkpoint(model, optimizer, epoch, path):
    """保存 FSDP checkpoint

    使用 FULL_STATE_DICT 模式: 将所有分片聚合到 rank 0 后保存。
    对于大模型，推荐使用 SHARDED_STATE_DICT 模式（见注释）。
    """
    save_policy = FullStateDictConfig(
        offload_to_cpu=True,  # 聚合后转到 CPU，节省 GPU 显存
        rank0_only=True,      # 只在 rank 0 上保留完整 state_dict
    )

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': None,  # 优化器状态也可以保存
            }, path)
            print(f'[Checkpoint] 已保存到 {path}')


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='FSDP 多模态对比预训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='每张 GPU 的 batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减')
    parser.add_argument('--strategy', type=str, default='full_shard',
                        choices=['full_shard', 'shard_grad_op', 'no_shard'],
                        help='FSDP 分片策略')
    parser.add_argument('--use_mp', action='store_true', default=True,
                        help='是否使用混合精度训练')
    parser.add_argument('--no_mp', action='store_false', dest='use_mp',
                        help='禁用混合精度')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint 保存目录')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每 N 个 epoch 保存一次 checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    # ---- 初始化分布式环境 ----
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # 设置随机种子（每个 rank 使用不同的种子以增加数据多样性）
    random.seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    if local_rank == 0:
        print(f'分布式训练: {world_size} 个 GPU')
        print(f'配置: {args}')

    # ---- 创建数据集 ----
    tokenizer = SimpleTokenizer(max_len=12)
    train_dataset = CIFAR10WithCaptions(
        root=args.data_root, train=True, tokenizer=tokenizer,
    )

    # DistributedSampler: 确保每个 GPU 处理不同的数据子集
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if local_rank == 0:
        print(f'词表大小: {tokenizer.vocab_size}')
        print(f'训练集: {len(train_dataset)} 样本')
        print(f'每 GPU batch size: {args.batch_size}')
        print(f'全局 batch size: {args.batch_size * world_size}')

    # ---- 创建模型并用 FSDP 包装 ----
    model = MiniCLIP(vocab_size=tokenizer.vocab_size)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'模型参数量: {total_params:,} ({total_params / 1e6:.1f}M)')

    model = setup_fsdp_model(model, local_rank, args)

    # ---- 优化器和调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # ---- 训练循环 ----
    os.makedirs(args.save_dir, exist_ok=True)

    if local_rank == 0:
        print(f'\n开始训练: {args.epochs} epochs\n')

    for epoch in range(1, args.epochs + 1):
        # 每个 epoch 重置 sampler，确保数据顺序不同
        train_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, epoch, local_rank,
        )
        scheduler.step()

        # 定期保存 checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = os.path.join(args.save_dir, f'mini_clip_epoch{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, save_path)

    if local_rank == 0:
        print('\n训练完成！')

    # ---- 清理分布式环境 ----
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
