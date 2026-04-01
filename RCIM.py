import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Basic convolution block with batch normalization and activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=False, activation='relu'):
        super(ConvBNAct, self).__init__()

        if activation == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation: {activation}')

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            act_layer
        )

    def forward(self, x):
        return self.block(x)


class CNN_trans(nn.Module):
    def __init__(self, img_size=320, inc=384, outc=384):
        super(CNN_trans, self).__init__()

        self.img_size = img_size
        self.conv0_f1 = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
        self.bn0_f1 = nn.BatchNorm2d(outc)

        self.conv0_f2 = nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1)
        self.bn0_f2 = nn.BatchNorm2d(outc)

    def forward(self, token_pair):
        """
        Args:
            token_pair: list or tuple containing paired token features
        Returns:
            tokens_ls: transformed token list
        """
        tokens_ls = []
        pair_num = len(token_pair) // 2

        for index in range(pair_num):
            f1 = F.relu(self.bn0_f1(self.conv0_f1(token_pair[index * 2])), inplace=True)
            f2 = F.relu(self.bn0_f2(self.conv0_f2(token_pair[index * 2 + 1])), inplace=True)
            tokens_ls.extend([f1, f2])

        return tokens_ls


class CAFF(nn.Module):
    def __init__(self, inc, outc, embed_dim):
        super(CAFF, self).__init__()

        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.bn_p2 = nn.BatchNorm2d(outc)

        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(
            nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, glbmap):
        """
        Args:
            x: local feature
            glbmap: global/context feature
        Returns:
            p2: fused output feature
        """
        x = torch.cat([x, glbmap], dim=1)
        x = self.conv_fuse(x)

        p1 = self.conv_p1(x)
        matt = self.sigmoid(p1)
        matt = matt * (1 - matt)
        matt = self.conv_matt(matt)

        fea = x * (1 + matt)

        p2 = self.conv_p2(fea)
        p2 = F.relu(self.bn_p2(p2), inplace=True)

        return p2


class Conv_Decoder(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(Conv_Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_chs, out_chs, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        return self.conv1(feats)


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=48, scale=4, dilation=None, aggregation=True):
        """
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of each split branch
            scale: number of branches
            dilation: dilation rate for each branch
            aggregation: whether to aggregate previous branch outputs
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = baseWidth
        self.nums = scale
        self.scale = scale
        self.aggregation = aggregation

        self.conv1 = nn.Conv2d(inplanes, self.width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dilation = [1] * self.nums if dilation is None else dilation

        for i in range(self.nums):
            self.convs.append(
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=3,
                    padding=dilation[i],
                    dilation=dilation[i],
                    bias=False
                )
            )
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, self.width, dim=1)
        out = None

        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]

            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), dim=1)

        out = self.bn3(self.conv3(out))
        out = out + x
        out = self.relu(out)

        return out


class Fusion(nn.Module):
    def __init__(self, channels=192, r=4):
        super(Fusion, self).__init__()

        inter_channels = int(channels // r)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.fu1 = ReceptiveConv(192, 192, baseWidth=48, dilation=[1, 2, 2, 4])
        self.fu2 = ReceptiveConv(384, 384, baseWidth=96, dilation=[1, 2, 2, 4])

    def forward(self, x, residual):
        xa = x + residual
        gwei = torch.sigmoid(self.global_att(xa))

        xg = x * gwei
        resig = residual * gwei

        xfuse = self.fu1(xg)
        xagg = torch.cat((xfuse, resig), dim=1)
        xo = self.fu2(xagg)

        return xo


class OutPut(nn.Module):
    def __init__(self, in_chs):
        super(OutPut, self).__init__()

        self.out = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, 1, 1)
        )

    def forward(self, feat):
        return self.out(feat)


class RCIMdecoder(nn.Module):
    def __init__(self):
        super(RCIMdecoder, self).__init__()

        self.trans_1 = CNN_trans(inc=384, outc=384)
        self.trans_2 = CNN_trans(inc=384, outc=384)

        self.caff_1 = CAFF(inc=384, outc=192, embed_dim=384)
        self.caff_2 = CAFF(inc=384, outc=192, embed_dim=384)

        self.conv_caff_fusion1 = Conv_Decoder(in_chs=192, out_chs=192)
        self.conv_caff_fusion2 = Conv_Decoder(in_chs=192, out_chs=192)

        self.conv_fusion_1 = Conv_Decoder(in_chs=384, out_chs=192)

        self.fusion1 = Fusion(channels=192)

        self.out1 = OutPut(in_chs=192)

    def forward(self, tokens, mode=None):
        """
        Args:
            tokens: token feature list
            mode: inference mode
        Returns:
            preds: prediction tuple
            out1: final output tensor
        """
        ################################################################
        cnn_trans_ls = [self.trans_1, self.trans_2]
        tokens_ls = []

        pair_num = len(tokens) // 2
        for index in range(pair_num):
            token_pair = [tokens[index * 2], tokens[index * 2 + 1]]
            token_pair_out = cnn_trans_ls[index](token_pair)
            tokens_ls.extend(token_pair_out)

        token_pair2 = [tokens[1], tokens[2]]
        token_pair_out2 = cnn_trans_ls[1](token_pair2)
        tokens_ls.extend(token_pair_out2)

        ################################################################
        feat2 = self.caff_1(tokens_ls[0], tokens_ls[1])
        feat1 = self.caff_2(tokens_ls[2], tokens_ls[3])

        ################################################################
        feat1 = self.conv_caff_fusion1(feat1)
        feat2 = self.conv_caff_fusion2(feat2)

        out1 = self.fusion1(feat1, feat2)
        feat1_1 = self.conv_fusion_1(out1)
        out1 = self.out1(feat1_1)

        _, _, size, _ = out1.shape
        out1 = F.interpolate(out1, size=(size * 8, size * 8), mode='bilinear', align_corners=False)

        ################################################################
        if mode == 'Test':
            preds = (out1,)
        else:
            out1 = torch.sigmoid(out1)
            out1 = torch.cat((1 - out1, out1), dim=1)
            preds = (out1,)

        return preds, out1