
from .base import BaseLoss, gather_and_scale_wrapper
import torch
import torch.nn as nn


class CenterLoss(BaseLoss):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        use_gpu (bool): whether to use GPU.
    """

    def __init__(self, num_classes=1024, feat_dim=256, use_gpu=True, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        n, c, p = embeddings.size()

        # 将 embeddings 转换为二维数组 [n*p, c]
        embeddings_reshaped = embeddings.permute(0, 2, 1).contiguous().view(-1, c).float()  # [n*p, c]

        # 获取新的批量大小
        batch_size = embeddings_reshaped.size(0)

        # 计算中心距离矩阵
        distmat = torch.pow(embeddings_reshaped, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, embeddings_reshaped, self.centers.t())
        distmat.addmm_(embeddings_reshaped, self.centers.t(), beta=1, alpha=-2)  # 使用新的参数签名

        # 创建类别标签张量
        labels = labels.repeat(p)
        classes = torch.arange(self.num_classes, device=embeddings.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)

        # 创建掩码
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # 计算损失
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss, self.info