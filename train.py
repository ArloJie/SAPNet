import argparse
import sys

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import *
from datasets import *
from RCIM import *
from loss import *
from metric import *


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--data_dir', default='./data/COD/', type=str)
parser.add_argument('--dataset', default='', type=str)
parser.add_argument('--valdataset', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=4, type=int)

###############################################################################
# Hyperparameters
###############################################################################
parser.add_argument('--max_epoch', default=40, type=int)
parser.add_argument('--optim', default='SGD', type=str, help='SGD or Adam')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--base_lr', default=0.0001, type=float)
parser.add_argument('--max_lr', default=0.01, type=float)
parser.add_argument('--wd', default=0.0005, type=float)
parser.add_argument('--momen', default=0.9, type=float)

###############################################################################
# Loss parameters
###############################################################################
parser.add_argument('--lscloss_weight', default=1, type=int)
parser.add_argument('--lscloss_xy', default=6, type=int)
parser.add_argument('--lscloss_rgb', default=0.1, type=float)
parser.add_argument('--lscloss_radius', default=5, type=int)

###############################################################################
# Logging
###############################################################################
parser.add_argument('--print_ratio', default=0.2, type=float)
parser.add_argument('--tag', default='', type=str)


def build_experiment_dirs(tag):
    """Create directories used in training and evaluation."""
    log_dir = create_directory('./experiments/logs/')
    data_dir = create_directory('./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{tag}/')

    log_path = log_dir + f'{tag}.txt'
    data_path = data_dir + f'{tag}.json'
    model_path = model_dir + f'{tag}.pth'

    pred_path = f'./experiments/images/{tag}'
    create_directory(pred_path)
    create_directory(pred_path + '/train')
    create_directory(pred_path + '/test')
    create_directory(pred_path + '/train/predictions')
    create_directory(pred_path + '/test/predictions')

    return log_path, data_path, model_dir, model_path, tensorboard_dir


def build_token_lists(token_tensor):
    """
    Convert token tensor into a list of layer-wise tokens.

    Args:
        token_tensor: [B, L, C, H, W]

    Returns:
        token_lists: list of tensors, each with shape [B, C, H, W]
    """
    token_lists = []
    for layer_idx in range(token_tensor.shape[1]):
        layer_token = torch.stack(
            [token_tensor[batch_idx, layer_idx] for batch_idx in range(token_tensor.shape[0])],
            dim=0
        )
        token_lists.append(layer_token)

    return token_lists


def build_final_affinity(token_tensor):
    """
    Build the affinity matrix from the last token layer.

    Args:
        token_tensor: [B, L, C, H, W]

    Returns:
        token_affinity: normalized affinity map
    """
    final_layer_token = []

    last_layer_idx = token_tensor.shape[1] - 1
    layer_token = torch.stack(
        [token_tensor[batch_idx, last_layer_idx] for batch_idx in range(token_tensor.shape[0])],
        dim=0
    )

    for batch_idx in range(layer_token.shape[0]):
        tokent = layer_token[batch_idx].view(384, 1600).transpose(1, 0)
        token_final = tokent @ tokent.transpose(0, 1)
        final_layer_token.append(token_final)

    final_layer_token = torch.stack(final_layer_token, dim=0)

    token_affinity = (final_layer_token - torch.min(final_layer_token)) / (
        torch.max(final_layer_token) - torch.min(final_layer_token) + 1e-8
    )

    return token_affinity


def evaluate_one_dataset(model, data_loader):
    """
    Evaluate model on one validation dataset.

    Args:
        model: validation model
        data_loader: dataloader for validation

    Returns:
        mae, meanf, sm, em, wfm
    """
    recorder = MetricRecorder(len(data_loader))
    loop = tqdm(data_loader, total=len(data_loader))

    for _, (images, token_tensor, gt) in enumerate(loop, 0):
        images = images.cuda()
        token_tensor = token_tensor.cuda()
        gt = gt.cuda()

        token_lists = build_token_lists(token_tensor)

        _, out1 = model(token_lists, mode='Test')

        pred = out1.sigmoid_().cpu().data.numpy()[0, 0]
        gt = gt.squeeze(0)
        gt = (gt > 0.5).cpu().numpy().astype(float)

        recorder.update(pre=pred, gt=gt)

    mae, (maxf, meanf, *_), sm, em, wfm = recorder.show(bit_num=3)
    return mae, meanf, sm, em, wfm


if __name__ == '__main__':
    args = parser.parse_args()

    ###########################################################################
    # Paths and logging
    ###########################################################################
    log_path, data_path, model_dir, model_path, tensorboard_dir = build_experiment_dirs(args.tag)

    log_func = lambda string='': log_print(string, log_path)
    log_func(f'[i] {args.tag}')
    log_func()

    ###########################################################################
    # Dataset
    ###########################################################################
    train_dataset = Train_Datasets(args.data_dir, args.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataset = Val_Datasets(args.data_dir, args.valdataset)
    val_loader = DataLoader(val_dataset, batch_size=1)

    train_iteration = len(train_loader)
    val_iteration = len(val_loader)
    log_iteration = int(train_iteration * args.print_ratio)
    max_iteration = args.max_epoch * train_iteration

    log_func(f'[i] train data : {len(train_dataset):,}')
    log_func(f'[i] log_iteration : {log_iteration:,}')
    log_func(f'[i] train_iteration : {train_iteration:,}')
    log_func(f'[i] max_iteration : {max_iteration:,}')

    ###########################################################################
    # Network
    ###########################################################################
    model = RCIMdecoder()
    model = model.cuda()
    model.train()

    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    load_model_fn = lambda: load_model(model, model_path)
    save_model_fn = lambda: save_model(model, model_path)

    ###########################################################################
    # Loss
    ###########################################################################
    lsc_kernels_desc_defaults = [{
        'weight': args.lscloss_weight,
        'xy': args.lscloss_xy,
        'rgb': args.lscloss_rgb
    }]
    lscloss_radius = args.lscloss_radius

    criterion = [
        MultiPBceloss().cuda(),
        LSalCoherenceloss().cuda(),
        Gsaloss().cuda(),
        Sscloss(0.85).cuda()
    ]

    ###########################################################################
    # Optimizer
    ###########################################################################
    decoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'directive filter' in name:
            head_params.append(param)
        else:
            decoder_params.append(param)

    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(
            [{'params': decoder_params}, {'params': head_params}],
            lr=args.lr,
            momentum=args.momen,
            weight_decay=args.wd,
            nesterov=True
        )
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
            [{'params': decoder_params, 'lr': args.base_lr}],
            betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f'Unsupported optimizer: {args.optim}')

    ###########################################################################
    # Training
    ###########################################################################
    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'pbce_loss', 'lsc_loss', 'gsa_loss', 'ssc_loss'])
    writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.max_epoch):
        loop = tqdm(train_loader, total=train_iteration)
        optimizer.zero_grad()

        for iteration, (images, token_tensor, cams) in enumerate(loop, 0):
            ###################################################################
            # Model input
            ###################################################################
            images = images.cuda()
            token_tensor = token_tensor.cuda()
            cams = cams.cuda()

            token_lists = build_token_lists(token_tensor)
            token_affinity = build_final_affinity(token_tensor)

            ###################################################################
            # Dynamic learning rate
            ###################################################################
            niter = epoch * train_iteration + (iteration - 1)
            lr, momentum = get_triangle_lr(
                args.base_lr,
                args.max_lr,
                args.max_epoch * train_iteration,
                niter,
                ratio=1.
            )
            optimizer.param_groups[0]['lr'] = lr
            optimizer.momentum = momentum

            ###################################################################
            # Forward
            ###################################################################
            pred, out1 = model(token_lists)

            scaled_token_lists = [
                F.interpolate(token, scale_factor=0.6, mode='bilinear', align_corners=False)
                for token in token_lists
            ]
            _, out1_s = model(scaled_token_lists)

            ###################################################################
            # Loss
            ###################################################################
            pbceloss = criterion[0](pred, cams)

            image_ = F.interpolate(images, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}

            out1_ = F.interpolate(out1[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            lscloss = criterion[1](
                out1_,
                lsc_kernels_desc_defaults,
                lscloss_radius,
                sample,
                image_.shape[2],
                image_.shape[3]
            )['loss']

            out1_t = F.interpolate(out1[:, 1:2], scale_factor=0.125, mode='bilinear', align_corners=True)
            gsaloss = 0.15 * criterion[2](out1_t, token_affinity)

            out2_scale = F.interpolate(out1[:, 1:2], scale_factor=0.6, mode='bilinear', align_corners=True)
            sscloss = criterion[3](out1_s[:, 1:2], out2_scale)

            loss = pbceloss + lscloss + gsaloss + sscloss

            ###################################################################
            # Optimization
            ###################################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            train_meter.add({
                'loss': loss.item(),
                'pbce_loss': pbceloss.item(),
                'lsc_loss': lscloss.item(),
                'gsa_loss': gsaloss.item(),
                'ssc_loss': sscloss.item(),
            })

            ###################################################################
            # Logging
            ###################################################################
            loop.set_description(f'Epoch: {epoch + 1}/{args.max_epoch}')

            if (iteration + 1) % 100 == 0:
                visualize_salicencymap(args.tag, images.clone().detach(), out1[:, 1:2], epoch, iteration)

                loss_avg, pbce_loss_avg, lsc_loss_avg, gsa_loss_avg, ssc_loss_avg = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch + 1,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss_avg,
                    'pbce_loss': pbce_loss_avg,
                    'lsc_loss': lsc_loss_avg,
                    'ssc_loss': ssc_loss_avg,
                    'gsa_loss': gsa_loss_avg,
                    'time': train_timer.tok(clear=True),
                }

                log_func('')
                log_func(
                    '[i]\t'
                    'Epoch[{epoch:,}/{max_epoch:,}],\t'
                    'iteration={iteration:,}, \t'
                    'learning_rate={learning_rate:.6f}, \t'
                    'loss={loss:.4f}, \t'
                    'pbce_loss={pbce_loss:.4f}, \t'
                    'lsc_loss={lsc_loss:.4f}, \t'
                    'ssc_loss={ssc_loss:.4f}, \t'
                    'gsa_loss={gsa_loss:.4f}, \t'
                    'time={time:.0f}sec'.format(**data)
                )

                writer.add_scalar('Train/loss', loss_avg, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #######################################################################
        # Evaluation
        #######################################################################
        if ((epoch + 1) < 31 and (epoch + 1) % 5 == 0) or ((epoch + 1) > 31 and (epoch + 1) % 1 == 0):
            with torch.no_grad():
                current_model_path = model_dir + f'{args.tag}-{epoch + 1}.pth'
                save_model(model, current_model_path)
                log_func('[i] save model')

                #################################################################
                # Validation model
                #################################################################
                val_model = RCIMdecoder()
                load_model(val_model, current_model_path)
                log_func('--------------------------------')
                log_func('[i] load model')

                val_model = val_model.cuda()
                val_model.eval()

                mae, meanf, sm, em, wfm = evaluate_one_dataset(val_model, val_loader)
                print('Mean-F: {}, EM: {}, MAE: {:.3f}, SM: {}'.format(meanf, em, round(mae, 3), sm))

                val_model = None
                torch.cuda.empty_cache()

    print(args.tag)