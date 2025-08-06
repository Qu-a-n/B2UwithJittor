import argparse
from datetime import datetime
import torch
import os
import logging
import numpy as np
from torchvision import transforms
from PIL import Image

from model.arch_unet import UNet
from model.b2u import (
    AugmentNoise,
    Masker
)
from model.utils import (
    setup_logger,
    validation_bsd300,
    validation_kodak,
    validation_Set14
)
from model.matric import (
    calculate_psnr,
    calculate_ssim
)



parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50'])
parser.add_argument('--checkpoint', type=str, default='./*.pth')
parser.add_argument('--test_dirs', type=str, default='./dataset/validation')
parser.add_argument('--save_test_path', type=str, default='./test')
parser.add_argument('--log_name', type=str, default='b2u_unet_g25_112rf20')
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument("--beta", type=float, default=20.0)

config, _ = parser.parse_known_args()
systime = datetime.now().strftime('%Y-%m-%d-%H-%M')
torch.set_num_threads(8)

os.makedirs(config.save_test_path, exist_ok=True)
setup_logger(
    "test",
    config.save_test_path,
    "test_" + config.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True
)
logger = logging.getLogger("test")

def save_model(model, epoch, name):
    save_path = os.path.join(config.save_test_path, 'models')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{}.pth'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_model(load_path, model, strict=True):
    assert load_path is not None, "FilePathError in load_model!"
    logger.info("Loading model from [{}] ...".format(load_path))
    load_net = torch.load(load_path)
    model.load_state_dict(load_net, strict=strict)
    return model



if __name__ == "__main__":
    # Validation Set
    Kodak_dir = os.path.join(config.test_dirs, "Kodak24")
    BSD300_dir = os.path.join(config.test_dirs, "BSD300")
    Set14_dir = os.path.join(config.test_dirs, "Set14")
    valid_dict = {
        "Kodak24": validation_kodak(Kodak_dir),
        "BSD300": validation_bsd300(BSD300_dir),
        "Set14": validation_Set14(Set14_dir)
    }

    # model tools 
    noiser = AugmentNoise(style=config.noisetype)
    masker = Masker(width=4, mode='interpolate', mask_type='all')
    model = UNet(
        in_channels=config.n_channel,
        out_channels=config.n_channel,
        wf=config.n_feature
    ).cuda()
    model = load_model(config.checkpoint, model, strict=True)
    beta = config.beta
    model.eval()

    # test
    save_test_path = os.path.join(config.save_test_path, config.log_name)
    validation_path = os.path.join(save_test_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)
    valid_repeat_times = {"Kodak24": 2, "BSD300": 1, "Set14": 2}

    for valid_name, valid_images in valid_dict.items():
        save_dir = os.path.join(validation_path, valid_name)
        os.makedirs(save_dir, exist_ok=True)
        logger.info('Processing {} dataset'.format(valid_name))
        avg_psnr_dn = []
        avg_ssim_dn = []
        avg_psnr_exp = []
        avg_ssim_exp = []
        avg_psnr_mid = []
        avg_ssim_mid = []
        repeat_times = valid_repeat_times[valid_name]
        for i in range(repeat_times):
            for idx, im in enumerate(valid_images):
                origin255 = im.copy()
                origin255 = origin255.astype(np.uint8)
                im = np.array(im, dtype=np.float32) / 255.0
                noisy_im = noiser.add_valid_noise(im)
                noisy255 = noisy_im.copy()
                noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)
                # padding to square
                H = noisy_im.shape[0]
                W = noisy_im.shape[1]
                val_size = (max(H, W) + 31) // 32 * 32
                noisy_im = np.pad(
                    noisy_im,
                    [[0, val_size - H], [0, val_size - W], [0, 0]],
                    'reflect')
                transformer = transforms.Compose([transforms.ToTensor()])
                noisy_im = transformer(noisy_im)
                noisy_im = torch.unsqueeze(noisy_im, 0)
                noisy_im = noisy_im.cuda()
                with torch.no_grad():
                    n, c, h, w = noisy_im.shape
                    net_input, mask = masker.train(noisy_im)
                    noisy_output = (model(net_input)*mask).view(n,-1,c,h,w).sum(dim=1)
                    exp_output = model(noisy_im)
                pred_dn = noisy_output[:, :, :H, :W]
                pred_exp = exp_output[:, :, :H, :W]
                pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)

                pred_dn = pred_dn.permute(0, 2, 3, 1)
                pred_exp = pred_exp.permute(0, 2, 3, 1)
                pred_mid = pred_mid.permute(0, 2, 3, 1)

                pred_dn = pred_dn.cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_exp = pred_exp.cpu().data.clamp(0, 1).numpy().squeeze(0)
                pred_mid = pred_mid.cpu().data.clamp(0, 1).numpy().squeeze(0)

                pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)
                pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)
                pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                    255).astype(np.uint8)                   

                # calculate psnr
                psnr_dn = calculate_psnr(origin255.astype(np.float32),
                                            pred255_dn.astype(np.float32))
                avg_psnr_dn.append(psnr_dn)
                ssim_dn = calculate_ssim(origin255.astype(np.float32),
                                            pred255_dn.astype(np.float32))
                avg_ssim_dn.append(ssim_dn)

                psnr_exp = calculate_psnr(origin255.astype(np.float32),
                                            pred255_exp.astype(np.float32))
                avg_psnr_exp.append(psnr_exp)
                ssim_exp = calculate_ssim(origin255.astype(np.float32),
                                            pred255_exp.astype(np.float32))
                avg_ssim_exp.append(ssim_exp)

                psnr_mid = calculate_psnr(origin255.astype(np.float32),
                                            pred255_mid.astype(np.float32))
                avg_psnr_mid.append(psnr_mid)
                ssim_mid = calculate_ssim(origin255.astype(np.float32),
                                            pred255_mid.astype(np.float32))
                avg_ssim_mid.append(ssim_mid)

                logger.info(
                    "{} - img:{}_{:03d} - PSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}; PSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}; PSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}.".format(
                    valid_name, i, idx, psnr_dn, ssim_dn, psnr_exp, ssim_exp, psnr_mid, ssim_mid
                    )
                )

                # visualization
                save_path = os.path.join(
                    save_dir,
                    "{}-{:03d}-{:03d}_clean.png".format(
                        valid_name, i, idx))
                Image.fromarray(origin255).convert('RGB').save(
                    save_path)
                save_path = os.path.join(
                    save_dir,
                    "{}-{:03d}-{:03d}_noisy.png".format(
                        valid_name, i, idx))
                Image.fromarray(noisy255).convert('RGB').save(
                    save_path)
                save_path = os.path.join(
                    save_dir,
                    "{}-{:03d}-{:03d}_dn.png".format(
                        valid_name, i, idx))
                Image.fromarray(pred255_dn).convert('RGB').save(save_path)
                save_path = os.path.join(
                    save_dir,
                    "{}-{:03d}-{:03d}_exp.png".format(
                        valid_name, i, idx))
                Image.fromarray(pred255_exp).convert('RGB').save(save_path)
                save_path = os.path.join(
                    save_dir,
                    "{}-{:03d}-{:03d}_mid.png".format(
                        valid_name, i, idx))
                Image.fromarray(pred255_mid).convert('RGB').save(save_path)

        avg_psnr_dn = np.array(avg_psnr_dn)
        avg_psnr_dn = np.mean(avg_psnr_dn)
        avg_ssim_dn = np.mean(avg_ssim_dn)

        avg_psnr_exp = np.array(avg_psnr_exp)
        avg_psnr_exp = np.mean(avg_psnr_exp)
        avg_ssim_exp = np.mean(avg_ssim_exp)

        avg_psnr_mid = np.array(avg_psnr_mid)
        avg_psnr_mid = np.mean(avg_psnr_mid)
        avg_ssim_mid = np.mean(avg_ssim_mid)
        
        logger.info(
            "----Average PSNR/SSIM results for {}----\n\tPSNR_DN: {:.6f} dB; SSIM_DN: {:.6f}\n----PSNR_EXP: {:.6f} dB; SSIM_EXP: {:.6f}\n----PSNR_MID: {:.6f} dB; SSIM_MID: {:.6f}".format(
                valid_name, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid
            )
        )

