import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss', 'DeepSupervisionBCEDiceLoss', 'compute_kl_loss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class DeepSupervisionBCEDiceLoss(nn.Module):
    def __init__(self, weights=[1, 1, 1, 1, 1]):
        super().__init__()
        self.base_loss = BCEDiceLoss()
        self.weights = weights

    def forward(self, outputs, targets):
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)

        total_loss = 0
        for output, weight in zip(outputs, self.weights):
            loss = self.base_loss(output, targets)
            total_loss += weight * loss
        return total_loss / len(self.weights)  # 平均加权损失

def compute_kl_loss(p, q):
    #用于计算两个概率分布之间的Kullback-Leibler散度（KL散度）
    #KL散度是衡量两个概率分布之间差异的一个重要指标
    #p: 第一个概率分布的预测值，通常是模型的输出
    #q: 第二个概率分布的目标值，通常是实际标签的分布。
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')#计算p到q的kl散度
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),#计算q到p的kl散度
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()#平均损失
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
#最终的损失是 p_loss 和 q_loss 的平均值


""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()