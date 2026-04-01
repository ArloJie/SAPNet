import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndi
import numpy as np


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

#  PBCEloss
class MultiPBceloss(nn.Module):
    def __init__(self):
        super(MultiPBceloss, self).__init__()
        self.pbce = nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')

    def forward(self, preds, cams):
        # 计算最后一个预测的 PBCE 损失

        bg_label = torch.full_like(cams, 255).long()
        fg_label = torch.full_like(cams, 0).long()
        bg_label[cams < 0.2] = 0
        fg_label[cams > 0.6] = 1

        processed_bg_labels = torch.zeros_like(bg_label)
        processed_fg_labels = torch.zeros_like(fg_label)
        for batch_idx in range(bg_label.shape[0]):

            single_bg = bg_label[batch_idx].squeeze(0).cpu().numpy()

            label, num_labels = ndi.label(single_bg)
            area_sizes = ndi.sum(single_bg, label, range(1, num_labels + 1))

            max_area = np.max(area_sizes)
            area_threshold = max_area / 2
            large_regions = area_sizes > area_threshold

            output_bg = np.zeros_like(single_bg)

            for region_idx in range(1, num_labels + 1):
                if large_regions[region_idx - 1]:  # 如果该区域大于阈值
                    output_bg[label == region_idx] = 255  # 将该区域标记为 255
            processed_bg_labels[batch_idx, 0] = torch.tensor(output_bg, dtype=torch.long)

        for batch_idx in range(fg_label.shape[0]):
            single_fg = fg_label[batch_idx].squeeze(0).cpu().numpy()
            label, num_labels = ndi.label(single_fg)
            area_sizes = ndi.sum(single_fg, label, range(1, num_labels + 1))
            max_area = np.max(area_sizes)
            area_threshold = max_area / 2
            large_regions = area_sizes > area_threshold
            output_fg = np.zeros_like(single_fg)
            for region_idx in range(1, num_labels + 1):
                if large_regions[region_idx - 1]:  # 如果该区域大于阈值
                    output_fg[label == region_idx] = 1  # 将该区域标记为 255

            output_fg = np.where(output_fg == 0, 255, output_fg)

            processed_fg_labels[batch_idx, 0] = torch.tensor(output_fg, dtype=torch.long)

        bg_label = processed_bg_labels.squeeze(1)
        fg_label = processed_fg_labels.squeeze(1)

        pbce_loss = 0.0
        # 遍历除最后一个预测以外的其他预测，按加权求 PBCE 损失
        for i in range(0, len(preds)):
            pbce_loss += (self.pbce(preds[i], fg_label) + self.pbce(preds[i], bg_label)) * ((1 ** i) / 1)

        return pbce_loss

#  LSCloss
class LSalCoherenceloss(nn.Module):
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input
    ):

        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
        )

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded) #滑窗产生的像素相加减

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))  #最终相乘

        out = {'loss': loss.mean(),}

        return out

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':   #关于距离的需要自己计算
                    feature = LSalCoherenceloss._get_mesh(N, height_pred, width_pred, device)
                else:          #这边就是RGB相关的特征处理，不需要自己计算（mesh）
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]

                feature /= sigma
                features.append(feature)
                # 上述计算出全图的特征
            features = torch.cat(features, dim=1)
            # 下述是计算领域中的像素
            kernel = weight * LSalCoherenceloss._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = LSalCoherenceloss._unfold(features, radius)      # 滑窗产生局部特征的过程
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)  # 目测是除去中心点，计算与周围点的方差
        #(N, C, diameter, diameter, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()  # kernel核 就是文中的F(i,j)
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),        #.repeat扩展
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)   #滑窗diameter * diameter的正方形

#  GSAloss
class Gsaloss(nn.Module):
    def __init__(self):
        super(Gsaloss, self).__init__()

    def forward(self, pred, token):

        B, C, H, W = pred.shape

        pred_1 = pred.reshape(B, H * W, 1)
        f_token = pred_1 * token  # A and V

        pred_2 = pred.reshape(B, 1, H * W)
        fore_A = f_token * pred_2

        b_token = (1 - pred_1) * token  # B and V
        back_B = b_token * (1 - pred_2)

        gsa_loss = 2 - torch.sum(fore_A) / torch.sum(f_token) - torch.sum(back_B) / torch.sum(b_token)

        return gsa_loss

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

class Sscloss(nn.Module):
    def __init__(self, alpha=0.85):
        """
        Saliency Structure Consistency Loss
        :param alpha: Weighting factor for SSIM and L1 loss
        """
        super(Sscloss, self).__init__()
        self.alpha = alpha

    def forward(self, x, y):
        ssim = torch.mean(SSIM(x, y))
        l1_loss = torch.mean(torch.abs(x - y))
        loss_ssc = self.alpha * ssim + (1 - self.alpha) * l1_loss
        return loss_ssc