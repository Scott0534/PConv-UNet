import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from dataloader.dataset import MedicalDataSets
from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score
import csv
from network.MSAA_Unet import MSAA_Unet
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

import datetime
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 关键新增行
# 配置参数类
class Config:
    def __init__(self):
        # 基础配置
        self.seed = 41

        self.model_name = " MSAA_Unet"

        self.img_size = 256
        self.max_epochs = 300

        # 路径配置
        self.base_dir = "./data/busi"
        # self.base_dir = "./data/BUS"
        self.train_file = "busi_train1337.txt"
        # self.train_file = "BUS_train1337.txt"
        self.val_file = "busi_val1337.txt"
        # self.val_file = "BUS_val1337.txt"
        self.checkpoint_dir = "./result"
        self.log_dir = "./result"

        # 训练参数
        self.batch_size = 8
        self.num_workers = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.min_lr = 1e-5


def seed_everything(seed):
    """设置全局随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(config, phase='train'):
    """获取数据增强组合"""
    if phase == 'train':
        return A.Compose([
            A.Resize(config.img_size, config.img_size),  # 统一尺寸
            # 几何变换
            A.Rotate(limit=(-20, 20), p=0.5),  # 随机旋转[-20°, 20°]
            A.Flip(p=0.5),  # 水平和垂直翻转
            A.ShiftScaleRotate(
                shift_limit=0.05,  # 5%平移
                scale_limit=0.1,  # 10%缩放
                rotate_limit=0,  # 禁用额外旋转
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            # 对比度增强（CLAHE更适合医学影像）
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),#用于对图像进行对比度限制自适应直方图均衡化
            # 亮度/对比度微调（±20%）
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.3
            ),
            # 添加超声斑点噪声（高斯+泊松混合）
            A.GaussNoise(var_limit=(10, 50), p=0.3),#高斯噪音，指定了方差范围，方差越大，噪声越多。
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),#ISO噪声在模拟相机在高ISO（感光度）下拍摄时的噪点效果。ISO噪声通常包括两种类型的噪声：色度噪声和亮度噪声。

            A.ElasticTransform(
                alpha=1,  # 形变强度
                sigma=2,  # 平滑系数
                alpha_affine=5,  # 仿射变换强度
                border_mode=cv2.BORDER_CONSTANT,
                p=0.1  # 低概率使用
            ),
            A.CoarseDropout(#（即随机遮挡图像的一部分）。这种技术可以帮助模型学习到对图像中局部缺失或遮挡的鲁棒性，尤其适用于医学图像处理，
                max_holes=2,#表示每个图像上最多可以创建2个矩形遮挡区域。
                max_height=0.2,  # 最大遮挡高度为20%图像
                max_width=0.2,#最大宽度可以是图像宽度的20%。
                min_holes=1,#表示每个图像上至少要创建1个矩形遮挡区域。
                fill_value=0,  # 填充黑色（模拟超声图像背景）
                mask_fill_value=None,#表示遮挡区域在掩码图像上不进行特定的填充，保持原有值
                p=0.2
            ),
            A.Normalize()
        ])
    return A.Compose([
        A.Resize(config.img_size, config.img_size),
        A.Normalize()
    ])


def create_dataloaders(config):
    """创建数据加载器"""
    train_ds = MedicalDataSets(
        base_dir=config.base_dir,
        split="train",
        transform=get_transforms(config, 'train'),
        train_file_dir=config.train_file,
        val_file_dir=config.val_file,

    )

    val_ds = MedicalDataSets(
        base_dir=config.base_dir,
        split="val",
        transform=get_transforms(config, 'val'),
        train_file_dir=config.train_file,
        val_file_dir=config.val_file,

    )

    print(f"训练集数量: {len(train_ds)}, 验证集数量: {len(val_ds)}")

    return (
        DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=False,

        ),
        DataLoader(
            val_ds,
            batch_size=4,
            shuffle=False,
            num_workers=4,

        )
    )


def initialize_model(config):
    """模型初始化"""
    model_registry = {
        " MSAA_Unet":  MSAA_Unet,
        # "CMUNet": CMUNet,
        # "AAUnet": AAUnet,
        # "Elsknet": Elsknet
    }

    return model_registry[config.model_name]().cuda()


def train_epoch(model, loader, criterion, optimizer):
    """训练单个epoch"""
    model.train()
    metrics = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter()
    }

    for batch in loader:
        images = batch['image'].cuda(non_blocking=True)
        masks = batch['label'].cuda(non_blocking=True)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新指标
        iou, dice, *_ = iou_score(outputs, masks)
        metrics['loss'].update(loss.item(), images.size(0))
        metrics['iou'].update(iou, images.size(0))
        metrics['dice'].update(dice, images.size(0))

    return metrics


def validate(model, loader, criterion):
    """验证过程"""
    model.eval()
    metrics = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'recall': AverageMeter(),
        'precision': AverageMeter(),
        'f1': AverageMeter(),
        'specificity': AverageMeter(),
        'acc': AverageMeter()
    }

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].cuda(non_blocking=True)
            masks = batch['label'].cuda(non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            iou, dice, recall, precision, f1, specificity, acc = iou_score(outputs, masks)

            metrics['loss'].update(loss.item(), images.size(0))
            metrics['iou'].update(iou, images.size(0))
            metrics['dice'].update(dice, images.size(0))
            metrics['recall'].update(recall, images.size(0))
            metrics['precision'].update(precision, images.size(0))
            metrics['f1'].update(f1, images.size(0))
            metrics['specificity'].update(specificity, images.size(0))
            metrics['acc'].update(acc, images.size(0))

    return metrics


def main(config):
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # 在初始化timestamp之后定义最佳模型路径
    best_model_name = f"{config.model_name}_best_{timestamp}.pth"
    best_model_path = os.path.join(config.checkpoint_dir, best_model_name)

    # 初始化环境
    seed_everything(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 初始化组件
    train_loader, val_loader = create_dataloaders(config)
    model = initialize_model(config)


    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs , eta_min=config.min_lr)

    criterion = losses.BCEDiceLoss().cuda()

    # 日志记录（修改文件名）
    csv_path = os.path.join(config.log_dir, f'training_log_{config.model_name}_{timestamp}.csv')
    best_iou = 0.0

    # 新增保存多个最佳 IoU 及其指标的列表
    num_best_metrics = 3  # 默认保存3个最佳指标，可根据需要调整
    best_metrics_history = []  # 保存多个最佳指标

    # 训练循环
    for epoch in range(config.max_epochs):
        start_time = time.time()

        # 训练阶段
        train_metrics = train_epoch(model, train_loader, criterion, optimizer)

        # 验证阶段
        val_metrics = validate(model, val_loader, criterion)



        scheduler.step()

        # 将当前验证指标加入到历史记录中
        current_iou = val_metrics['iou'].avg
        current_metrics = {k: v.avg for k, v in val_metrics.items()}
        current_metrics['epoch'] = epoch

        # 更新最佳指标历史记录
        # 保留最多 `num_best_metrics` 个最佳指标
        best_metrics_history = sorted(
            best_metrics_history + [current_metrics],
            key=lambda x: x['iou'],
            reverse=True
        )[:num_best_metrics]

        # 记录日志
        log_data = [
            epoch + 1,
            optimizer.param_groups[0]['lr'],
            train_metrics['loss'].avg,
            train_metrics['iou'].avg,
            train_metrics['dice'].avg,
            val_metrics['loss'].avg,
            val_metrics['iou'].avg,
            val_metrics['dice'].avg,
            val_metrics['recall'].avg,
            val_metrics['precision'].avg,
            val_metrics['f1'].avg,
            val_metrics['specificity'].avg,
            val_metrics['acc'].avg,
        ]

        # 写入CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['Epoch', 'LR', 'Train Loss', 'Train IoU', 'Train Dice',
                                 'Val Loss', 'Val IoU', 'Val Dice', 'Recall', 'Precision',
                                 'F1', 'Specificity', 'Accuracy'])
            writer.writerow(log_data)

        # 保存最佳模型
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"🚀 更新最佳模型: {os.path.abspath(best_model_path)}")

        # 打印训练信息
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{config.max_epochs} | 耗时: {epoch_time:.1f}s")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        print("训练指标:")
        print(
            f"  Loss: {train_metrics['loss'].avg:.4f} | IoU: {train_metrics['iou'].avg:.4f} | Dice: {train_metrics['dice'].avg:.4f}")
        print("验证指标:")
        print(
            f"  Loss: {val_metrics['loss'].avg:.4f} | IoU: {val_metrics['iou'].avg:.4f} | Dice: {val_metrics['dice'].avg:.4f}")
        print("详细指标:")
        print(f"  Recall:    {val_metrics['recall'].avg:.4f}")
        print(f"  Precision: {val_metrics['precision'].avg:.4f}")
        print(f"  F1:        {val_metrics['f1'].avg:.4f}")
        print(f"  Specificity: {val_metrics['specificity'].avg:.4f}")
        print(f"  Accuracy:  {val_metrics['acc'].avg:.4f}")
        print("-" * 80)

        print(f"更新最佳历史记录 (前 {num_best_metrics} 个)：")
        for metrics in best_metrics_history:
            print(metrics)

    # 最终输出
    print("\n🔥 训练完成!")
    print(f"最佳模型路径: {os.path.abspath(best_model_path)}")
    print(f"最佳验证IoU: {best_iou:.4f}")

    # 初始化变量
    last_three_epochs = []

    # 尝试读取最后三轮的数据
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) <= 1:  # 确保有数据
                print("CSV日志数据不足，无法计算最后三轮")
            else:
                data_rows = rows[1:]  # 跳过表头
                last_three_rows = data_rows[-3:]  # 取最后三行

                for row in last_three_rows:
                    # 解析每行数据，假设列顺序与写入时一致
                    epoch_num, lr, train_loss, train_iou, train_dice, \
                        val_loss, val_iou, val_dice, recall, precision, \
                        f1, specificity, acc = row
                    last_three_epochs.append({
                        'iou': float(val_iou),
                        'dice': float(val_dice),
                        'recall': float(recall),
                        'precision': float(precision),
                        'f1': float(f1),
                        'specificity': float(specificity),
                        'acc': float(acc)
                    })
    except FileNotFoundError:
        print(f"未找到日志文件: {csv_path}")

    # 打印最后三轮的指标
    if last_three_epochs:
        print("\n最后三轮指标：")
        for metrics in last_three_epochs:
            print(metrics)

    # 计算最后三轮指标的均值和标准差
    if last_three_epochs:
        ious = [m['iou'] for m in last_three_epochs]
        dice = [m['dice'] for m in last_three_epochs]
        recall = [m['recall'] for m in last_three_epochs]
        precision = [m['precision'] for m in last_three_epochs]
        f1 = [m['f1'] for m in last_three_epochs]
        specificity = [m['specificity'] for m in last_three_epochs]
        acc = [m['acc'] for m in last_three_epochs]

        iou_mean = np.mean(ious)
        iou_std = np.std(ious)
        dice_mean = np.mean(dice)
        dice_std = np.std(dice)
        recall_mean = np.mean(recall)
        recall_std = np.std(recall)
        precision_mean = np.mean(precision)
        precision_std = np.std(precision)
        f1_mean = np.mean(f1)
        f1_std = np.std(f1)
        specificity_mean = np.mean(specificity)
        specificity_std = np.std(specificity)
        acc_mean = np.mean(acc)
        acc_std = np.std(acc)

        print("\n最后三轮指标的均值和标准差:")
        print(f"IoU:        {iou_mean:.8f} ± {iou_std:.8f}")
        print(f"Dice:       {dice_mean:.8f} ± {dice_std:.8f}")
        print(f"Recall:     {recall_mean:.8f} ± {recall_std:.8f}")
        print(f"Precision:  {precision_mean:.8f} ± {precision_std:.8f}")
        print(f"F1:         {f1_mean:.8f} ± {f1_std:.8f}")
        print(f"Specificity: {specificity_mean:.8f} ± {specificity_std:.8f}")
        print(f"Accuracy:   {acc_mean:.8f} ± {acc_std:.8f}")

        # 保存到 CSV 文件
        summary_path = os.path.join(config.log_dir, f'last_three_metrics_summary_{timestamp}.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean', 'Std'])
            writer.writerow(['IoU', f"{iou_mean:.8f}", f"{iou_std:.8f}"])
            writer.writerow(['Dice', f"{dice_mean:.8f}", f"{dice_std:.8f}"])
            writer.writerow(['Recall', f"{recall_mean:.8f}", f"{recall_std:.8f}"])
            writer.writerow(['Precision', f"{precision_mean:.8f}", f"{precision_std:.8f}"])
            writer.writerow(['F1', f"{f1_mean:.8f}", f"{f1_std:.8f}"])
            writer.writerow(['Specificity', f"{specificity_mean:.8f}", f"{specificity_std:.8f}"])
            writer.writerow(['Accuracy', f"{acc_mean:.8f}", f"{acc_std:.8f}"])


if __name__ == "__main__":
    config = Config()
    main(config)
