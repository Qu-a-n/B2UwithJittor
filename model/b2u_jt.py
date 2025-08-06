import jittor as jt
import numpy as np
from jittor.dataset import Dataset, DataLoader
from typing import Any, List, Dict, Tuple
import glob
import os
from PIL import Image


operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    np.random.seed(operation_seed_counter)
    return operation_seed_counter


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
            std = std * jt.ones((shape[0], 1, 1, 1))
            noise = jt.randn(shape) * std
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = jt.rand(shape[0], 1, 1, 1) * (max_std - min_std) + min_std
            noise = jt.randn(shape) * std
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * jt.ones((shape[0], 1, 1, 1))
            noised = jt.array(np.random.poisson(lam.numpy() * x.numpy())) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = jt.rand(shape[0], 1, 1, 1) * (max_lam - min_lam) + min_lam
            noised = jt.array(np.random.poisson(lam.numpy() * x.numpy())) / lam
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
    return jt.nn.PixelShuffle(block_size)(x)

def gen_mask(img: jt.Var, width=4, mask_type='random'):
    """
    generate random mask for each patch
    """
    n, c, h, w = img.shape
    mask = jt.zeros((n * (h // width) * (w // width) * width**2, ), dtype=jt.int64)
    idx_list = jt.arange(0, width**2, 1, dtype=jt.int64)
    rand_idx = jt.zeros((n * h // width * w // width, ), dtype=jt.int64)

    if mask_type == 'random':
        rand_idx = jt.randint(
            low=0,
            high=len(idx_list),
            shape=(n * (h // width) * (w // width), )
        )

    elif mask_type == 'batch':
        pass

    elif mask_type == 'all':
        rand_idx = jt.randint(
            low=0,
            high=len(idx_list),
            shape=(1, )
        ).repeat(n * h // width * w // width)

    elif 'fix' in mask_type:
        index = mask_type.split("_")[-1]
        index = jt.array(np.array(index).astype(np.int64))
        rand_idx = index.repeat(n * (h // width) * (w // width))
    else:
        raise ValueError("InValid maskType!")

    rand_pair_idx = idx_list[rand_idx]
    rand_pair_idx += jt.arange(
        start=0,
        end=n * (h // width) * (w // width) * width**2,
        step=width**2,
        dtype=jt.int64
    ) # get absolute index
    mask[rand_pair_idx] = 1
    mask = depth2space(
        mask.float().view(n, h // width, w // width, width**2).permute(0,3,1,2),
        block_size=width
    ).int()

    return mask

def interpolate_mask(tensor: jt.Var, mask, mask_inv):
    """
    interpolate mask_idx value by conv2d with width**2 kernel
    """
    n, c, h, w = tensor.shape
    kernel = np.array([
        [0.5, 1.0, 0.5],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 0.5]
    ])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = kernel / kernel.sum()
    kernel = jt.array(kernel, dtype=jt.float32)
    
    filter_tensor = jt.nn.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=1
    )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask_inv.ndim == 3:
        mask_inv = mask_inv.unsqueeze(1)
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
    
    def mask(self, img, mask_type=None, mode=None) -> Tuple[jt.Var, jt.Var]:
        mode = self.mode if mode is None else mode
        mask_type = self.mask_type if mask_type is None else mask_type
        assert mode == 'interpolate', "NotImplementedError!"

        n, c, h, w = img.shape
        mask = gen_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = jt.ones_like(mask) - mask
        masked = interpolate_mask(img, mask, mask_inv)

        net_input = masked
        return net_input, mask
    
    def train(self, img) -> Tuple:
        n, c, h, w = img.shape
        tensors = jt.zeros((n, self.width**2, c, h, w))
        masks = jt.zeros((n, self.width**2, 1, h, w))
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
        print(f"Processing file {fn}")
        img = Image.open(fn)
        print(f"Original image mode: {img.mode}, size: {img.size}")
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        img = np.array(img, dtype=np.float32)
        print(f"Numpy array shape: {img.shape}")
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
            print(f"Converted grayscale to 3-channel: {img.shape}")
        # random crop patch
        H = img.shape[0]
        W = img.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            img = img[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            img = img[:, yy:yy + self.patch, :]
        # Convert to jittor tensor
        img = jt.array(img).permute(2, 0, 1)  # HWC to CHW
        return img


    def __len__(self):
        return len(self.train_fns)
    
def create_dataloader_train(
    dir,
    patch=256,
    num_workers=8,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    max_samples=400
) -> DataLoader:
    
    TrainDataset = Dataset4Train(dir, patch, max_samples)
    return DataLoader(
        dataset=TrainDataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
