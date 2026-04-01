import os
import cv2
import torch
import random
import time
import cmapy
import numpy as np

from torchvision import utils
import torch.nn.functional as F


def create_directory(path):
    """Create a directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_txt(path, string):
    """Append one line of text to a file."""
    with open(path, "a+", encoding="utf-8") as f:
        f.write(string + "\n")


def log_print(message, path):
    """Print a message and save it to a log file."""
    print(message)
    add_txt(path, message)


def load_model(model, model_path, parallel=False, map_location=None):
    """Load model weights."""
    state_dict = torch.load(model_path, map_location=map_location)
    if parallel:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def save_model(model, model_path, parallel=False):
    """Save model weights."""
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def get_learning_rate_from_optimizer(optimizer):
    """Get the current learning rate from the first parameter group."""
    return optimizer.param_groups[0]["lr"]


def check_positive(am):
    """
    Check whether the activation map is dominated by positive responses
    on image borders.
    """
    am = am.clone()
    am[am > 0.5] = 1
    am[am <= 0.5] = 0

    edge_mean = (
        am[0, 0, 0, :].mean()
        + am[0, 0, :, 0].mean()
        + am[0, 0, -1, :].mean()
        + am[0, 0, :, -1].mean()
    ) / 4

    return edge_mean > 0.5


def calculate_parameters(model):
    """Calculate the number of parameters in millions."""
    return sum(param.numel() for param in model.parameters()) / 1_000_000.0


class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.tik()

    def tik(self):
        """Start timing."""
        self.start_time = time.time()

    def tok(self, ms=False, clear=False):
        """
        Stop timing and return elapsed time.

        Args:
            ms: return milliseconds if True, otherwise seconds
            clear: restart timer after reading
        """
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.tik()

        return duration


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        """Add a dictionary of scalar values."""
        for key, value in dic.items():
            if key in self.data_dic:
                self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        """Get the average values of recorded metrics."""
        if keys is None:
            keys = self.keys

        dataset = []
        for key in keys:
            values = self.data_dic[key]
            dataset.append(float(np.mean(values)) if len(values) > 0 else 0.0)

        if clear:
            self.clear()

        if len(dataset) == 1:
            return dataset[0]

        return dataset

    def clear(self):
        """Clear all stored metric values."""
        self.data_dic = {key: [] for key in self.keys}


def _tensor_to_bgr_image(image_tensor):
    """Convert a normalized tensor image to uint8 BGR image."""
    grid = utils.make_grid(
        image_tensor.unsqueeze(0),
        nrow=1,
        padding=0,
        pad_value=0,
        normalize=True
    )
    image = (
        grid.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to('cpu', torch.uint8)
        .numpy()[..., ::-1]
    )
    return image


def _normalize_attmap(attmap):
    """Normalize an attention map to [0, 255]."""
    attmap = attmap / (np.max(attmap) + 1e-5)
    attmap = np.uint8(attmap * 255)
    return attmap


def _build_overlay(image, attmap, out_size):
    """Create a colored overlay image from attention map and RGB image."""
    colormap = cv2.applyColorMap(
        cv2.resize(attmap, out_size),
        cmapy.cmap('seismic')
    )
    cam = colormap + 0.4 * image
    cam = cam / (np.max(cam) + 1e-5)
    cam = np.uint8(cam * 255).copy()
    return cam


def visualize_heatmap(experiments, images, attmaps, epoch, cnt, phase='train'):
    """Visualize and save heatmaps."""
    _, _, h, w = images.shape
    attmaps = attmaps.squeeze().to('cpu').detach().numpy()

    save_dir = f'./labels/images/{experiments}/{phase}/colormaps'
    create_directory(save_dir)

    for i in range(images.shape[0]):
        attmap = attmaps[i]
        attmap = _normalize_attmap(attmap)

        image = _tensor_to_bgr_image(images[i])
        cam = _build_overlay(image, attmap, (w, h))

        cv2.imwrite(f'{save_dir}/{epoch}-{cnt}-{i}-image.jpg', image)
        cv2.imwrite(f'{save_dir}/{epoch}-{cnt}-{i}-colormap.jpg', cam)


def visualize_salicencymap(experiments, images, attmaps, epoch, cnt, phase='train'):
    """Visualize and save saliency maps."""
    _, _, h, w = images.shape
    attmaps = attmaps.squeeze().to('cpu').detach().numpy()

    save_dir = f'./experiments/images/{experiments}/{phase}/predictions'
    create_directory(save_dir)

    for i in range(images.shape[0]):
        attmap = attmaps[i]

        binarymap = (attmap - np.min(attmap)) / (np.max(attmap) - np.min(attmap) + 1e-5)
        binarymap = ((binarymap > 0.5) * 255).astype(np.uint8)
        binarymap = cv2.resize(binarymap, (320, 320), interpolation=cv2.INTER_NEAREST)

        norm_attmap = _normalize_attmap(attmap)
        image = _tensor_to_bgr_image(images[i])
        cam = _build_overlay(image, norm_attmap, (w, h))

        cv2.imwrite(f'{save_dir}/{epoch}-{cnt}-{i}-image.jpg', image)
        cv2.imwrite(f'{save_dir}/{epoch}-{cnt}-{i}-colormap.jpg', cam)
        cv2.imwrite(f'{save_dir}/{epoch}-{cnt}-{i}-salmap.jpg', binarymap)


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        """
        SGD optimizer with polynomial learning rate decay.
        """
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

        self.global_step = 0
        self.max_step = max_step
        self.poly_power = momentum
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.poly_power

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1


def get_strided_size(orig_size, stride):
    """Get the strided size for a given input size and stride."""
    return ((orig_size[0] - 1) // stride + 1, (orig_size[1] - 1) // stride + 1)


def get_strided_up_size(orig_size, stride):
    """Get the upsampled strided size."""
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0] * stride, strided_size[1] * stride


def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    """Resize tensors with interpolation."""
    return F.interpolate(tensors, size=size, mode=mode, align_corners=align_corners)


def make_cam(x, epsilon=1e-5):
    """Generate normalized CAM from raw activation."""
    x = F.relu(x)

    b, c, h, w = x.size()
    flat_x = x.view(b, c, h * w)
    max_value = flat_x.max(dim=-1)[0].view(b, c, 1, 1)

    return F.relu(x - epsilon) / (max_value + epsilon)


def get_numpy_from_tensor(tensor):
    """Convert a tensor to numpy array."""
    return tensor.cpu().detach().numpy()


def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    """Apply a color map to a CAM image."""
    if shape is not None:
        h, w, _ = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam, cmapy.cmap('seismic'))
    return cam


def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1.0,
                    annealing_decay=1e-2, momentums=(0.95, 0.85)):
    """
    Compute triangular learning rate and momentum.
    """
    first = int(total_steps * ratio)
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur / total_steps)
    x = np.abs(cur * 2.0 / total_steps - 2.0 * cycle + 1)

    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0.0, 1.0 - x)
    else:
        lr = ((base_lr - min_lr) * cur + min_lr * first - base_lr * total_steps) / (first - total_steps)

    if isinstance(momentums, (int, float)):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0.0, 1.0 - x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    """Compute polynomial decayed learning rate."""
    return base_lr * (1.0 - min(last_epoch, num_steps - 1) / num_steps) ** power