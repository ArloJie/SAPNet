import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from numpy import load


# General function to get file paths for different dataset phases (train, val, test, etc.)
def get_file_paths(data_dir, name, phase, type='rgb', suffix='.npy', gt_suffix='.png', cam_suffix='cam.png'):
    """Fetch paths for images, tokens, and ground truths based on the dataset phase."""
    name_list = []
    image_root = os.path.join(data_dir, name, 'images')
    img_list = sorted(os.listdir(image_root))

    token_root = os.path.join(data_dir, name, 'tokens')
    gt_root = os.path.join(data_dir, name, 'segmentations') if 'val' in phase or 'test' in phase else ''
    cam_root = os.path.join(data_dir, name, 'train-cam') if 'train' in phase else ''

    for img_name in img_list:
        img_tag = img_name.split('.')[0]
        tag_dict = {
            'rgb': os.path.join(image_root, img_name),
            'token': os.path.join(token_root, img_tag + suffix),
        }
        if 'val' in phase or 'test' in phase:
            tag_dict['gt'] = os.path.join(gt_root, img_tag + gt_suffix)
        if 'train' in phase:
            tag_dict['cam'] = os.path.join(cam_root, img_tag + cam_suffix)

        name_list.append(tag_dict)
    return name_list


# Data transformation for normalization and conversion to tensor
ToTensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Retaining the original `Train_label_Datasets` and `Test_label_Datasets` classes
class Train_label_Datasets(Dataset):
    """Training dataset class that includes labels and tokens."""

    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset
        self.img_list = get_file_paths(self.data_dir, self.dataset, phase='train', suffix='.npy', gt_suffix='.png',
                                       cam_suffix='cam.png')

    def __getitem__(self, index):
        sample_path = self.img_list[index]
        img_path = sample_path['rgb']
        token_path = sample_path['token']  # Token path

        image = Image.open(img_path).convert('RGB')
        token = load(token_path)  # Load token

        image = image.resize((320, 320), resample=Image.LANCZOS)
        image = ToTensor(image)

        token_tensor = torch.from_numpy(token)
        token_tensor = token_tensor.permute(0, 2, 1)  # Adjust dimensions
        token_tensor = token_tensor[2, :, :].view(384, 40, 40)  # Reshape

        return image, token_tensor

    def __len__(self):
        return len(self.img_list)


class Test_label_Datasets(Dataset):
    """Testing dataset class that includes labels and tokens."""

    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset
        self.img_list = get_file_paths(self.data_dir, self.dataset, phase='test', suffix='.npy', gt_suffix='.png')

    def __getitem__(self, index):
        sample_path = self.img_list[index]
        img_path = sample_path['rgb']
        token_path = sample_path['token']  # Token path
        gt_path = sample_path['gt']

        image = Image.open(img_path).convert('RGB')
        token = load(token_path)  # Load token
        image_name = os.path.basename(img_path)
        gt = Image.open(gt_path).convert('L')

        image = image.resize((320, 320), resample=Image.LANCZOS)
        tensor_image = ToTensor(image)

        token_tensor = torch.from_numpy(token)
        token_tensor = token_tensor.permute(0, 2, 1)  # Adjust dimensions
        token_tensor = token_tensor[2, :, :].view(384, 40, 40)  # Reshape

        gt = gt.resize((320, 320), Image.NEAREST)
        gt = np.array(gt, dtype=np.float32)
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)

        return image, token_tensor, image_name, tensor_image, torch.from_numpy(gt)

    def __len__(self):
        return len(self.img_list)


class Train_Datasets(Dataset):
    """Training dataset class that includes images, tokens, and cam."""

    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset
        self.img_list = get_file_paths(self.data_dir, self.dataset, phase='train', suffix='.npy', gt_suffix='.png',
                                       cam_suffix='cam.png')

    def __getitem__(self, index):
        sample_path = self.img_list[index]
        img_path = sample_path['rgb']
        token_path = sample_path['token']  # Token path
        cam_path = sample_path['cam']

        image = Image.open(img_path).convert('RGB')
        token_tensor = load(token_path)
        cam = Image.open(cam_path).convert('L')

        image = image.resize((320, 320), resample=Image.LANCZOS)
        image = ToTensor(image)

        token_tensor = torch.from_numpy(token_tensor)
        token_tensor = token_tensor.permute(0, 2, 1)
        token_tensor = token_tensor.view(3, 384, 40, 40)

        cam = np.array(cam, dtype=np.float32)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-5)
        cam = np.expand_dims(cam, axis=0)

        return image, token_tensor, torch.from_numpy(cam)

    def __len__(self):
        return len(self.img_list)


class Val_Datasets(Dataset):
    """Validation dataset class that includes images, tokens, and ground truth."""

    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset
        self.img_list = get_file_paths(self.data_dir, self.dataset, phase='val', suffix='.npy', gt_suffix='.png')

    def __getitem__(self, index):
        sample_path = self.img_list[index]
        img_path = sample_path['rgb']
        token_path = sample_path['token']  # Token path
        gt_path = sample_path['gt']

        image = Image.open(img_path).convert('RGB')
        token_tensor = load(token_path)
        gt = Image.open(gt_path).convert('L')

        image = image.resize((320, 320), resample=Image.LANCZOS)
        image = ToTensor(image)

        token_tensor = torch.from_numpy(token_tensor)
        token_tensor = token_tensor.permute(0, 2, 1)
        token_tensor = token_tensor.view(3, 384, 40, 40)

        gt = gt.resize((320, 320), Image.NEAREST)
        gt = np.array(gt, dtype=np.float32)
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)

        return image, token_tensor, torch.from_numpy(gt)

    def __len__(self):
        return len(self.img_list)


class Test_Datasets(Dataset):
    """Testing dataset class that includes images, tokens, and ground truth."""

    def __init__(self, data_dir, dataset):
        self.data_dir = data_dir
        self.dataset = dataset
        self.img_list = get_file_paths(self.data_dir, self.dataset, phase='test', suffix='.npy', gt_suffix='.png')

    def __getitem__(self, index):
        sample_path = self.img_list[index]
        img_path = sample_path['rgb']
        token_path = sample_path['token']  # Token path
        gt_path = sample_path['gt']

        image = Image.open(img_path).convert('RGB')
        image_path = os.path.basename(img_path)
        token_tensor = load(token_path)
        gt = Image.open(gt_path).convert('L')

        image = image.resize((320, 320), resample=Image.LANCZOS)
        image = ToTensor(image)

        token_tensor = torch.from_numpy(token_tensor)
        token_tensor = token_tensor.permute(0, 2, 1)
        token_tensor = token_tensor.view(3, 384, 40, 40)

        gt = gt.resize((320, 320), Image.NEAREST)
        gt = np.array(gt, dtype=np.float32)
        gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-5)

        return image, token_tensor, torch.from_numpy(gt), image_path

    def __len__(self):
        return len(self.img_list)
