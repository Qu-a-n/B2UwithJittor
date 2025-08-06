import argparse
import os
import logging
import jittor as jt
import numpy as np
from PIL import Image
from typing import Tuple
import time


from datetime import datetime
from model.utils import (
    setup_logger,
    validation_kodak,
    validation_bsd300,
    validation_Set14
)
from model.arch_unet_jt import UNet
from model.b2u_jt import (
    AugmentNoise,
    Masker,
    create_dataloader_train,
)
from model.matric import (
    calculate_psnr, 
    calculate_ssim
)


parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25", choices=['gauss25', 'gauss5_50', 'poisson30', 'poisson5_50'])
parser.add_argument('--log_name', type=str, 
                default='b2u_unet_gauss25_112rf20')
parser.add_argument('--save_model_path', type=str,
                default='./results')
parser.add_argument('--train_dir', type=str,
                default='./dataset/train')
parser.add_argument('--validation_dir', type=str, 
                default='./dataset/validation')
parser.add_argument('--resume', type=str)
parser.add_argument('--checkpoint', type=str)

parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=2)  # 减少批次大小
parser.add_argument('--patchsize', type=int, default=128)
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=3)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=2.0)
parser.add_argument("--increase_ratio", type=float, default=20.0)
parser.add_argument('--name', type=str, default="model")


config, _ = parser.parse_known_args()
systime = datetime.now().strftime('%Y-%m-%d-%H-%M')
config.save_path = os.path.join(config.save_model_path, config.log_name, systime)

os.makedirs(config.save_path, exist_ok=True)
setup_logger(
    "train",
    config.save_path,
    "train_" + config.log_name,
    level=logging.INFO,
    screen=True,
    tofile=True
)
logger = logging.getLogger("train")


def save_state(epoch, optimizer, scheduler):
    """Saves training state during training, which will be used for resuming"""
    save_path = os.path.join(config.save_path, 'training_states')
    os.makedirs(save_path, exist_ok=True)
    state = {
        "epoch": epoch, 
        "scheduler": None,  # Jittor scheduler doesn't have state_dict
        "optimizer": optimizer.state_dict()
    }
    save_filename = "{}.state".format(epoch)
    save_path = os.path.join(save_path, save_filename)
    jt.save(state, save_path)

def resume_state(load_path, optimizer, scheduler) -> Tuple[int, jt.optim.Adam, jt.lr_scheduler.MultiStepLR]:
    """Resume the optimizers and schedulers for training"""
    resume_state = jt.load(load_path)
    epoch = resume_state["epoch"]
    resume_optimizer = resume_state["optimizer"]
    # resume_scheduler = resume_state["scheduler"]
    optimizer.load_state_dict(resume_optimizer)
    # scheduler.load_state_dict(resume_scheduler)
    return epoch, optimizer, scheduler

def save_model(model, epoch, name):
    save_path = os.path.join(config.save_path, 'models')
    os.makedirs(save_path, exist_ok=True)
    model_name = 'epoch_{}_{}.pkl'.format(name, epoch)
    save_path = os.path.join(save_path, model_name)
    state_dict = model.state_dict()
    jt.save(state_dict, save_path)
    logger.info('Checkpoint saved to {}'.format(save_path))


def load_model(load_path, model, strict=True):
    assert load_path is not None, "FilePathError in load_model!"
    logger.info("Loading model from [{}] ...".format(load_path))
    load_net = jt.load(load_path)
    model.load_state_dict(load_net)
    return model


if __name__ == "__main__":
    # prepare dataset
    trainLoader = create_dataloader_train(
        dir=config.train_dir,
        patch=config.patchsize,
        batch_size=config.batchsize,
        num_workers=2,
        max_samples=400
    )
    ## Validation dataset
    Kodak24_dir = os.path.join(config.validation_dir, "Kodak24")
    BSD300_dir = os.path.join(config.validation_dir, "BSD300")
    Set14_dir = os.path.join(config.validation_dir, "Set14")

    valid_dict = {
        "Kodak24": validation_kodak(Kodak24_dir),
        "BSD300": validation_bsd300(BSD300_dir),
        "Set14": validation_Set14(Set14_dir)
    }

    # -----------------------------------------------------

    # prepare training tools
    noiser = AugmentNoise(style=config.noisetype)
    masker = Masker(
        width=4,
        mode="interpolate",
        mask_type='all'
    )
    model = UNet(
        in_channels=config.n_channel,
        out_channels=config.n_channel,
        wf=config.n_feature
    )
    epochs = config.n_epoch
    ratio = epochs / 100
    optimizer = jt.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-8)
    scheduler = jt.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(20 * ratio) - 1,
            int(40 * ratio) - 1,
            int(60 * ratio) - 1,
            int(80 * ratio) - 1
        ],
        gamma=0.5
    )
    print("Batchsize={}, number of epoch={}".format(config.batchsize, config.n_epoch))
    
    ## if resume or reload
    epoch_bg = 1
    if config.resume:
        epoch_bg, optimizer, scheduler = resume_state(
            config.resume,
            optimizer,
            scheduler
        )
    if config.checkpoint:
        model = load_model(config.checkpoint, model, strict=True)
    lr_bg = scheduler.get_lr()
    logger.info('----------------------------------------------------')
    logger.info("==> Resuming Training with learning rate:{}".format(lr_bg))
    logger.info('----------------------------------------------------')
    print("Epoch_begin={} with learning_rate={}".format(epoch_bg, lr_bg))
    print("Init Finish." + '-'*50)

    if config.noisetype in ['gauss25', 'poisson30']:
        Thread1, Thread2 = 0.8, 1.0
    else:
        Thread1, Thread2 = 0.4, 1.0
    
    Lambda1, Lambda2 = config.Lambda1, config.Lambda2
    increase_ratio = config.increase_ratio

    # ------------------------------------------------

    
    # BEGIN Training ---------------------------------
    for epoch in range(1 if epoch_bg == 1 else epoch_bg + 1, epochs + 1):
        cnt = 0
        model.train()
        epoch_loss_all = []

        for i, clean in enumerate(trainLoader):
            start_time = time.time()
            clean = clean / 255.0
            noisy = noiser.add_train_noise(clean)
            
            # clean cache
            if i % 10 == 0:
                jt.clean()

            optimizer.zero_grad()

            nn_input, mask = masker.train(noisy)
            noisy_output = model(nn_input)
            n, c, h, w = noisy.shape
            noisy_output = (noisy_output*mask).view(n, -1, c, h, w).sum(dim=1)
            diff = noisy_output - noisy

            with jt.no_grad():
                exp_output = model(noisy)
            exp_diff = exp_output - noisy

            Lambda = epoch / epochs
            if Lambda <= Thread1:
                beta = Lambda2
            elif Thread1 <= Lambda <= Thread2:
                beta = Lambda2 + (Lambda - Thread1) * (increase_ratio-Lambda2) / (Thread2-Thread1)
            else:
                beta = increase_ratio
            alpha = Lambda1

            revisible = diff + beta * exp_diff
            loss_reg = alpha * jt.mean(diff**2)
            loss_rev = jt.mean(revisible**2)
            loss_all = loss_reg + loss_rev
            epoch_loss_all.append(loss_all.data)
            
            optimizer.backward(loss_all)
            optimizer.step()

            logger.info(
            '{:04d} {:05d} diff={:.6f}, exp_diff={:.6f}, Loss_Reg={:.6f}, Lambda={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'
            .format(epoch, i, jt.mean(diff**2).item(), jt.mean(exp_diff**2).item(),
                    loss_reg.item(), Lambda, loss_rev.item(), loss_all.item(), time.time() - start_time))
            
        scheduler.step()
        mean_loss_all = np.mean(epoch_loss_all)
        loss_log = os.path.join(config.save_path, "train_" + config.log_name + "_lossLog.log")
        with open(loss_log, 'a+') as f:
            f.write(f"Epoch {epoch} mean Loss_All: {mean_loss_all:.6f}")
            print(f"Epoch {epoch} mean Loss_All: {mean_loss_all:.6f}")

        if epoch % config.n_snapshot == 0 or epoch == epochs:
            model.eval()
            save_model(model, epoch, config.name)
            save_state(epoch, optimizer, scheduler)

            if epoch == epochs:
                # validation
                save_model_path = os.path.join(config.save_model_path, config.log_name,
                                           systime)
                validation_path = os.path.join(save_model_path, "validation")
                os.makedirs(validation_path, exist_ok=True)
                np.random.seed(101)
                valid_repeat_times = {"Kodak24": 2, "BSD300": 1, "Set14": 2}

                for valid_name, valid_images in valid_dict.items():
                    avg_psnr_dn = []
                    avg_ssim_dn = []
                    avg_psnr_exp = []
                    avg_ssim_exp = []
                    avg_psnr_mid = []
                    avg_ssim_mid = []

                    save_dir = os.path.join(validation_path, valid_name)
                    os.makedirs(save_dir, exist_ok=True)
                    repeat_times = valid_repeat_times[valid_name]

                    for t in range(repeat_times):
                        for idx, img in enumerate(valid_images):
                            if(idx % 4 == 0):
                                print(f"valid_name={valid_name}, repeat_time={t}, idx={idx}")
                            origin255 = img.copy()
                            origin255 = origin255.astype(np.uint8)

                            im = np.array(img, dtype=np.float32) / 255.0
                            noisy_im = noiser.add_valid_noise(im)
                            if epoch == config.n_snapshot:
                                noisy255 = noisy_im.copy()
                                noisy255 = np.clip(noisy255 * 255.0 + 0.5, 0,
                                                255).astype(np.uint8)

                            H = noisy_im.shape[0]
                            W = noisy_im.shape[1]
                            val_size = (max(H, W) + 31) // 32 * 32
                            noisy_im = np.pad(
                                noisy_im,
                                [[0, val_size - H], [0, val_size - W], [0, 0]],
                                'reflect'
                            )
                            noisy_im = jt.array(noisy_im).permute(2, 0, 1).unsqueeze(0)

                            with jt.no_grad():
                                n, c, h, w = noisy_im.shape
                                nn_input, mask = masker.train(noisy_im)
                                noisy_output = (model(nn_input) * mask).view(n, -1, c, h, w).sum(dim=1)
                                dn_output = noisy_output.detach().clone()

                                del nn_input, mask, noisy_output
                                jt.clean()
                                exp_output = model(noisy_im)
                            pred_dn = dn_output[:, :, :H, :W]
                            pred_exp = exp_output.detach().clone()[:, :, :H, :W]
                            pred_mid = (pred_dn + beta*pred_exp) / (1 + beta)

                            del exp_output
                            jt.clean()

                            pred_dn = pred_dn.permute(0, 2, 3, 1)
                            pred_exp = pred_exp.permute(0, 2, 3, 1)
                            pred_mid = pred_mid.permute(0, 2, 3, 1)

                            pred_dn = pred_dn.clamp(0, 1).numpy().squeeze(0)
                            pred_exp = pred_exp.clamp(0, 1).numpy().squeeze(0)
                            pred_mid = pred_mid.clamp(0, 1).numpy().squeeze(0)

                            pred255_dn = np.clip(pred_dn * 255.0 + 0.5, 0,
                                             255).astype(np.uint8)
                            pred255_exp = np.clip(pred_exp * 255.0 + 0.5, 0,
                                                255).astype(np.uint8)
                            pred255_mid = np.clip(pred_mid * 255.0 + 0.5, 0,
                                                255).astype(np.uint8)

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

                            if t == 0 and epoch == config.n_snapshot:
                                save_path = os.path.join(
                                    save_dir,
                                    "{}_{:03d}-{:03d}_clean.png".format(
                                        valid_name, idx, epoch))
                                Image.fromarray(origin255).convert('RGB').save(
                                    save_path)
                                save_path = os.path.join(
                                    save_dir,
                                    "{}_{:03d}-{:03d}_noisy.png".format(
                                        valid_name, idx, epoch))
                                Image.fromarray(noisy255).convert('RGB').save(
                                    save_path)
                            if t == 0:
                                save_path = os.path.join(
                                    save_dir,
                                    "{}_{:03d}-{:03d}_dn.png".format(
                                        valid_name, idx, epoch))
                                Image.fromarray(pred255_dn).convert(
                                    'RGB').save(save_path)
                                save_path = os.path.join(
                                    save_dir,
                                    "{}_{:03d}-{:03d}_exp.png".format(
                                        valid_name, idx, epoch))
                                Image.fromarray(pred255_exp).convert(
                                    'RGB').save(save_path)
                                save_path = os.path.join(
                                    save_dir,
                                    "{}_{:03d}-{:03d}_mid.png".format(
                                        valid_name, idx, epoch))
                                Image.fromarray(pred255_mid).convert(
                                    'RGB').save(save_path)

                    avg_psnr_dn = np.array(avg_psnr_dn)
                    avg_psnr_dn = np.mean(avg_psnr_dn)
                    avg_ssim_dn = np.mean(avg_ssim_dn)

                    avg_psnr_exp = np.array(avg_psnr_exp)
                    avg_psnr_exp = np.mean(avg_psnr_exp)
                    avg_ssim_exp = np.mean(avg_ssim_exp)

                    avg_psnr_mid = np.array(avg_psnr_mid)
                    avg_psnr_mid = np.mean(avg_psnr_mid)
                    avg_ssim_mid = np.mean(avg_ssim_mid)

                    log_path = os.path.join(validation_path,
                                            "A_log_{}.csv".format(valid_name))
                    with open(log_path, "a") as f:
                        f.writelines("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}\n".format(
                            epoch, avg_psnr_dn, avg_ssim_dn, avg_psnr_exp, avg_ssim_exp, avg_psnr_mid, avg_ssim_mid))
