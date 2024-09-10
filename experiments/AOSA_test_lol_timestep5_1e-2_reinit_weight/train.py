import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)


def get_trainer(predict_res_or_noise=None):
    # init
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
            sampling_timesteps = 2
        sampling_timesteps_original_ddim_ddpm = 250
        train_num_steps = 80000

    original_ddim_ddpm = False
    if original_ddim_ddpm:
        condition = False
        input_condition = False
        input_condition_mask = False
    else:
        condition = False
        input_condition = False
        input_condition_mask = False

    condition = True
    input_condition = False
    input_condition_mask = False

    if condition:
        # Image restoration  
        if input_condition:
            folder = ["/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                    "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                    "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                    "/home/liu/disk12t/liu_data/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                    "/home/liu/disk12t/liu_data/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                    "/home/liu/disk12t/liu_data/shadow_removal_with_val_dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
        else:
            folder = ["/home/liu/disk12t/liu_data/dataset/lol_datset/high_train.flist",
                    "/home/liu/disk12t/liu_data/dataset/lol_datset/low_train.flist",
                    "/home/liu/disk12t/liu_data/dataset/lol_datset/high_test.flist",
                    "/home/liu/disk12t/liu_data/dataset/lol_datset/low_test.flist", ]
        train_batch_size = 1
        num_samples = 1
        sum_scale = 1
        image_size = 256
    else:
        # Image Generation 
        folder = 'xxx/CelebA/img_align_celeba'
        train_batch_size = 128
        num_samples = 64
        sum_scale = 1
        image_size = 64

    num_unet = 1
    objective = 'auto_res_noise'
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
            loss_type='l1',            # L1 or L2
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise = test_res_or_noise,
            alpha_res_to_0_or_1=predict_res_or_noise
        )

    trainer = Trainer(
        diffusion,
        folder,
        train_batch_size=train_batch_size,
        num_samples=num_samples,
        train_lr=8e-5,
        train_num_steps=train_num_steps,         # total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=False,                        # turn on mixed precision
        convert_image_to="RGB",
        condition=condition,
        save_and_sample_every=save_and_sample_every,
        equalizeHist=True,
        crop_patch=False,
        generation=True,
        num_unet=num_unet,
    )
    
    return trainer, train_num_steps//save_and_sample_every

def main():
    # init
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
    sys.stdout.flush()
    set_seed(10)
    

    # train
    trainer,model_id = get_trainer(predict_res_or_noise=None)
    predict_res_or_noise = trainer.train(ATDP=True)
    if predict_res_or_noise:
        print("Residual prediction for this task.")
    else:
        print("Noise prediction for this task.")

    predict_res_or_noise = 1
    trainer,model_id = get_trainer(predict_res_or_noise=predict_res_or_noise)
    trainer.train(ATDP=False)


    # test
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        trainer.load(model_id)
        trainer.set_results_folder('./results/'+str(model_id))
        trainer.test(last=True)


if __name__ == '__main__':
    main()


