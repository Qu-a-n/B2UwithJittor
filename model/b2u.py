import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Any, List, Dict, Tuple
import glob
import os
from PIL import Image
from torchvision import transforms


from model.arch_unet import UNet
from model.matric import calculate_psnr, calculate_ssim


operation_seed_counter = 0

def get_generator(device="cuda"):
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device=device)
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


class AugmentNoise:

    def __init__(self, style):
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"
        else:
            raise ValueError("Invalid NoiseType!")


    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.empty(shape, dtype=torch.float32, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         generator=get_generator(),
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.empty(shape, dtype=torch.float32, device=x.device)
            torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised
        
    
    def add_valid_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32)
        elif self.style == "poisson_fix":
            lam = self.params[0]
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)

def depth2space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)

def gen_mask(img: torch.Tensor, width=4, mask_type='random'):
    """
    generate random mask for each patch
    """
    n, c, h, w = img.shape
    mask = torch.zeros(
        size=(n * (h // width) * (w // width) * width**2, ),
        dtype=torch.int64,
        device=img.device
    )
    idx_list = torch.arange(
        0, width**2, 1, dtype=torch.int64, device=img.device
    )
    rand_idx = torch.zeros(
        size=(n * h // width * w // width, ),
        dtype=torch.int64,
        device=img.device
    )

    if mask_type == 'random':
        torch.randint(
            low=0,
            high=len(idx_list),
            size=(n * (h // width) * (w // width), ),
            device=img.device,
            generator=get_generator(device=img.device),
            out=rand_idx
        )

    elif mask_type == 'batch':
        pass

    elif mask_type == 'all':
        rand_idx = torch.randint(
            low=0,
            high=len(idx_list),
            size=(1, ),
            device=img.device,
            generator=get_generator(device=img.device)
        ).repeat(n * h // width * w // width)

    elif 'fix' in mask_type:
        index = mask_type.split("_")[-1]
        index = torch.from_numpy(
            np.array(index).astype(np.int64)
        ).type(torch.int64)

        rand_idx = index.repeat(n * (h // width) * (w // width)).to(img.device)
    else:
        raise ValueError("InValid maskType!")

    rand_pair_idx = idx_list[rand_idx]
    rand_pair_idx += torch.arange(
        start=0,
        end=n * (h // width) * (w // width) * width**2,
        step=width**2,
        dtype=torch.int64,
        device=img.device
    ) # get absolute index
    mask[rand_pair_idx] = 1
    mask = depth2space(
        mask.type_as(img).view(n, h // width, w // width, width**2).permute(0,3,1,2),
        block_size=width
    ).type(torch.int64)

    return mask

def interpolate_mask(tensor: torch.Tensor, mask, mask_inv):
    """
    interpolate mask_idx value by conv2d with width**2 kernel
    """
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    kernel = np.array([
        [0.5, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 0.5]
    ])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = kernel / kernel.sum()
    kernel = torch.from_numpy(kernel).float().to(device)
    
    filter_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1
    )
    return filter_tensor.view_as(tensor) * mask + tensor * mask_inv


class Masker:

    def __init__(
            self,
            width=4,
            mode="interpolate",
            mask_type="all"
    ):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type
    
    def mask(self, img, mask_type=None, mode=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = self.mode if mode is None else mode
        mask_type = self.mask_type if mask_type is None else mask_type
        assert mode == 'interpolate', "NotImplementedError!"

        n, c, h, w = img.shape
        mask = gen_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape).to(img.device) - mask
        masked = interpolate_mask(img, mask, mask_inv)

        net_input = masked
        return net_input, mask
    
    def train(self, img) -> Tuple:
        n, c, h, w = img.shape
        tensors = torch.zeros(
            (n, self.width**2, c, h, w), device=img.device
        )
        masks = torch.zeros(
            (n, self.width**2, 1, h, w), device=img.device
        )
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


class Dataset4Train(Dataset):

    def __init__(self, dir, patch=256, max_samples=400):
        super().__init__()
        self.data_dir = dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, '*'))
        self.train_fns.sort()
        self.train_fns = self.train_fns[:max_samples]
        print('fetch {} samples for training (limited to first {} images)'.format(len(self.train_fns), max_samples))

    
    def __getitem__(self, index):
        fn = self.train_fns[index]
        img = Image.open(fn)
        img = np.array(img, dtype=np.float32)
        # random crop patch
        H = img.shape[0]
        W = img.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            img = img[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            img = img[:, yy:yy + self.patch, :]
        toTensor = transforms.Compose([transforms.ToTensor()])
        return toTensor(img)


    def __len__(self):
        return len(self.train_fns)
    
def create_dataloader_train(
    dir,
    patch=256,
    num_workers=8,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    pin_memory=False,
    max_samples=400
) -> DataLoader:
    
    TrainDataset = Dataset4Train(dir, patch, max_samples)
    return DataLoader(
        dataset=TrainDataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory
    )
