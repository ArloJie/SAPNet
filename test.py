import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import *
from datasets import *
from RCIM import *
from metric import *


parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--data_dir', default='./data/COD/', type=str)
parser.add_argument('--dataset', default='', type=str)
parser.add_argument('--num_workers', default=4, type=int)

###############################################################################
# Model and logging
###############################################################################
parser.add_argument('--tag', default='COD', type=str)
parser.add_argument('--model_name', default='-.pth', type=str)
parser.add_argument('--save_predictions', action='store_true')


def build_token_lists(token_tensor):
    """
    Convert token tensor into a list of layer-wise tokens.

    Args:
        token_tensor: Tensor with shape [B, L, C, H, W]

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


def save_prediction(pred, save_dir, image_path):
    """
    Save binary prediction mask.

    Args:
        pred: numpy array with shape [H, W], values in [0, 1]
        save_dir: output directory
        image_path: original image path list from dataloader
    """
    pred_map = out1.squeeze(0).squeeze(0).detach().cpu().numpy()
    pred_mask = ((pred_map > 0.5) * 255).astype(np.uint8)
    image_id = image_path[0].split('.')[0]
    Image.fromarray(pred_mask).convert('L').save(f'{save_dir}/{image_id}.png')


if __name__ == '__main__':
    args = parser.parse_args()

    ###########################################################################
    # Device
    ###########################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###########################################################################
    # Paths
    ###########################################################################
    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + args.model_name

    pred_root = create_directory(f'./vis-cam/{args.tag}/')
    pred_save_path = create_directory(f'./vis-cam/{args.tag}/predictions/')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    log_func = lambda string='': print(string)
    log_func(f'[i] {args.tag}')
    log_func(f'[i] model path: {model_path}')
    log_func(f'[i] device: {device}')
    log_func()

    ###########################################################################
    # Dataset
    ###########################################################################
    test_dataset = Test_Datasets(args.data_dir, args.dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False
    )

    test_iteration = len(test_loader)

    ###########################################################################
    # Network
    ###########################################################################
    model = RCIMdecoder().to(device)
    model.eval()

    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    ###########################################################################
    # Metrics
    ###########################################################################
    MR = MetricRecorder(test_iteration)
    total_inference_time = 0.0
    total_frames = 0

    ###########################################################################
    # Test
    ###########################################################################
    loop = tqdm(test_loader, total=test_iteration)

    with torch.no_grad():
        for _, (images, token_tensor, gt, image_path) in enumerate(loop, 0):
            ###################################################################
            # Model input
            ###################################################################
            images = images.to(device)
            token_tensor = token_tensor.to(device)
            gt = gt.to(device)

            token_lists = build_token_lists(token_tensor)

            ###################################################################
            # Forward and timing
            ###################################################################
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            _, out1 = model(token_lists, mode='Test')

            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            total_frames += 1

            loop.set_description('test')

            ###################################################################
            # Metrics
            ###################################################################
            pred = out1.sigmoid().detach().cpu().numpy()[0, 0]
            gt_np = gt.squeeze(0)
            gt_np = (gt_np > 0.5).cpu().numpy().astype(float)

            MR.update(pre=pred, gt=gt_np)

            ###################################################################
            # Save prediction if needed
            ###################################################################
            if args.save_predictions:
                save_prediction(out1, pred_save_path, image_path)

    ###########################################################################
    # Final metrics
    ###########################################################################
    mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
    print(
        'Mean-F: {}, EM: {}, MAE: {:.3f}, SM: {}, maxf: {}, wfm: {}'.format(
            meanf, em, round(mae, 3), sm, maxf, wfm
        )
    )

    real_fps = total_frames / total_inference_time if total_inference_time > 0 else 0.0
    print(f'[Result] FPS: {real_fps:.2f}')