import torch


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5  # 修正GT处理
    corr = torch.sum(SR == GT)
    tensor_size = SR.numel()
    acc = (corr.float() / tensor_size).item()
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5  # 修正GT处理
    TP = (SR & GT).sum().float()
    FN = (~SR & GT).sum().float()
    sensitivity = TP / (TP + FN + 1e-6)
    return sensitivity.item()


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5  # 修正GT处理
    TN = (~SR & ~GT).sum().float()
    FP = (SR & ~GT).sum().float()
    specificity = TN / (TN + FP + 1e-6)
    return specificity.item()


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > 0.5  # 修正GT处理
    TP = (SR & GT).sum().float()
    FP = (SR & ~GT).sum().float()
    precision = TP / (TP + FP + 1e-6)
    return precision.item()


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output)  # 确保输出在0-1之间

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum().float()
    union = (output_ | target_).sum().float()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * iou) / (iou + 1.0)  # 维持原Dice计算方式

    # 计算其他指标时直接使用二值化的output_和target_
    recall = get_sensitivity(output_, target_)
    precision = get_precision(output_, target_)
    specificity = get_specificity(output_, target_)
    acc = get_accuracy(output_, target_)
    F1 = 2 * recall * precision / (recall + precision + 1e-6)

    return (iou.item(), dice.item(), recall, precision, F1, specificity, acc)

# def iou_score(output_tuple, target, threshold=0.5):
#     smooth = 1e-5
#     iou_list = []
#     dice_list = []
#     recall_list = []
#     precision_list = []
#     f1_list = []
#     specificity_list = []
#     acc_list = []
#
#     # 遍历元组中的每个输出
#     for output in output_tuple:
#         # 确保输出是张量
#         if not torch.is_tensor(output):
#             raise TypeError("Each element in the output tuple must be a tensor.")
#
#         # 应用 sigmoid 和阈值处理
#         output_process = torch.sigmoid(output)  # 确保输出在0-1之间
#         output_ = output_process > threshold
#         target_ = target > 0.5  # 修正GT处理
#
#         # 计算IoU和Dice
#         intersection = (output_ & target_).sum().float()
#         union = (output_ | target_).sum().float()
#         iou = (intersection + smooth) / (union + smooth)
#         dice = (2.0 * iou) / (iou + 1.0)  # 维持原Dice计算方式
#
#         # 计算其他指标
#         TP = (output_ & target_).sum().float()
#         FN = (~output_ & target_).sum().float()
#         FP = (output_ & ~target_).sum().float()
#         TN = (~output_ & ~target_).sum().float()
#
#         recall = TP / (TP + FN + 1e-6)
#         precision = TP / (TP + FP + 1e-6)
#         F1 = 2 * recall * precision / (recall + precision + 1e-6) if (recall + precision) != 0 else 0.0
#         specificity = TN / (TN + FP + 1e-6)
#         acc = (TP + TN) / (TP + TN + FP + FN + 1e-6)
#
#         # 将结果添加到列表中
#         iou_list.append(iou.item())
#         dice_list.append(dice.item())
#         recall_list.append(recall.item())
#         precision_list.append(precision.item())
#         f1_list.append(F1)
#         specificity_list.append(specificity.item())
#         acc_list.append(acc.item())
#
#     # 计算每个指标的平均值
#     avg_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0.0
#     avg_dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0
#     avg_recall = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0.0
#     avg_precision = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0.0
#     avg_specificity = sum(specificity_list) / len(specificity_list) if len(specificity_list) > 0 else 0.0
#     avg_acc = sum(acc_list) / len(acc_list) if len(acc_list) > 0 else 0.0
#     avg_f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0.0
#
#     return (avg_iou, avg_dice, avg_recall, avg_precision, avg_f1, avg_specificity, avg_acc)