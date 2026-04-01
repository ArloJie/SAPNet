import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Basic Conv-BN-ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBN(nn.Module):
    """Basic Conv-BN block."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class FeatTrans(nn.Module):
    """Transform input token features from 384 channels to 128 channels."""
    def __init__(self, in_channels, out_channels):
        super(FeatTrans, self).__init__()
        self.convit = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bnvit = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tokens):
        trans_img = self.relu(self.bnvit(self.convit(tokens)))
        return trans_img


class BranchBlock(nn.Module):
    """One branch used in SimEncoder."""
    def __init__(self, in_channels):
        super(BranchBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        return self.block(x)


class EnhanceBlock(nn.Module):
    """
    Enhancement block:
    conv -> sigmoid gating -> gate*(1-gate) -> conv-bn-relu -> residual scaling -> conv-bn-relu
    """
    def __init__(self, channels):
        super(EnhanceBlock, self).__init__()
        self.conv_p1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv_matt = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn_p2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        att = torch.sigmoid(self.conv_p1(x))
        att = att * (1 - att)
        att = self.conv_matt(att)
        feat = x * (1 + att)
        out = self.conv_p2(feat)
        out = F.relu(self.bn_p2(out), inplace=True)
        return out


class SimEncoder(nn.Module):
    def __init__(self, in_channels):
        super(SimEncoder, self).__init__()

        # Three parallel branches
        self.left_branch = BranchBlock(in_channels)
        self.right_branch = BranchBlock(in_channels)
        self.add_branch = BranchBlock(in_channels)

        # Three branch-wise enhancement blocks
        self.enhance_left = EnhanceBlock(128)
        self.enhance_right = EnhanceBlock(128)
        self.enhance_add = EnhanceBlock(128)

        # Middle fusion branch after concatenation
        self.middle_branch = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=1),
            nn.BatchNorm2d(384)
        )

        # Final enhancement block
        self.enhance_middle = EnhanceBlock(384)

    def forward(self, features):
        left_out = self.left_branch(features)
        right_out = self.right_branch(features)
        add_out = self.add_branch(features)

        p21 = self.enhance_left(left_out)
        p22 = self.enhance_right(right_out)
        p23 = self.enhance_add(add_out)

        concatenated = torch.cat([p21, p22, p23], dim=1)
        middle_out = self.middle_branch(concatenated)
        p24 = self.enhance_middle(middle_out)

        return p24


class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        n, c, h, w = x.size()

        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(n, 1, h * w)
        x = x.reshape(n, c, h * w).permute(0, 2, 1).contiguous()

        fg_feats = torch.matmul(ccam_, x) / (h * w)
        bg_feats = torch.matmul(1 - ccam_, x) / (h * w)

        return fg_feats.reshape(n, -1), bg_feats.reshape(n, -1), ccam


class SDRDnet(nn.Module):
    def __init__(self):
        super(SDRDnet, self).__init__()

        self.trans_img = FeatTrans(in_channels=384, out_channels=128)
        self.SimEncoder = SimEncoder(128)
        self.ac_head = Disentangler(384)

        # Mark modules that should use the "from scratch" learning rate
        self.from_scratch_module_names = {
            'ac_head.activation_head',
        }

    def forward(self, token, inference=False):
        dino_img = self.trans_img(token)
        sim_out = self.SimEncoder(dino_img)
        fg_feats, bg_feats, ccam = self.ac_head(sim_out, inference)
        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        """
        Return parameter groups:
        groups[0]: pretrained weights
        groups[1]: pretrained biases
        groups[2]: from-scratch weights
        groups[3]: from-scratch biases
        """
        groups = ([], [], [], [])

        print('======================================================')

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.weight is not None and module.weight.requires_grad:
                    if name in self.from_scratch_module_names:
                        groups[2].append(module.weight)
                    else:
                        groups[0].append(module.weight)

                if module.bias is not None and module.bias.requires_grad:
                    if name in self.from_scratch_module_names:
                        groups[3].append(module.bias)
                    else:
                        groups[1].append(module.bias)

        return groups