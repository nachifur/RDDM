import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)
debug = False

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 100000

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = False
    input_condition = False
    input_condition_mask = False

if condition:
    # Image restoration  
    if input_condition:
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
    else:
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist"]
    train_batch_size = 1
    num_samples = 1
    sum_scale = 0.01
    image_size = 256
else:
    # Image Generation 
    folder = 'xxx/CelebA/img_align_celeba'
    train_batch_size = 128
    num_samples = 64
    sum_scale = 1
    image_size = 64

num_unet = 2
objective = 'pred_res_noise'
test_res_or_noise = "res_noise"
if original_ddim_ddpm:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise = test_res_or_noise
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l2',            # L1 or L2
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise = test_res_or_noise
    )

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=True,
    num_unet=num_unet,
)

# train
trainer.train()

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(trainer.train_num_steps//save_and_sample_every)
    trainer.set_results_folder(
        './results/test_timestep_'+str(sampling_timesteps))
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)
