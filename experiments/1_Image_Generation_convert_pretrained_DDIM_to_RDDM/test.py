import argparse
import glob
import os
import random

import numpy as np
import torch


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--seed", type=int, default=10, help="Random seed"
    )
    parser.add_argument(
        "--save_path", type=str, default="./RDDM_results", help="Path for saving running related data."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=10
    )
    parser.add_argument(
        "--to_RDDM", type=int, default=1
    )
    parser.add_argument(
        "--model_id", type=str, default="google/ddpm-celebahq-256"
    )  # "google/ddpm-cifar10-32"
    parser.add_argument(
        "--ratio", type=float, default=-1
    )
    parser.add_argument(
        "--adjust_alpha", type=int, default=0
    )
    parser.add_argument(
        "--schedule", type=str, default="decreased"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    args = parser.parse_args()
    return args


def main():
    # init
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(e) for e in [args.gpu_id])
    set_seed(args.seed)
    num_inference_steps = args.num_inference_steps
    ratio = args.ratio
    to_RDDM = True if args.to_RDDM==1 else False
    schedule = args.schedule
    model_id = args.model_id
    batch_size = args.batch_size
    adjust_alpha = True if args.adjust_alpha==1 else False
    model_name = "RDDM" if to_RDDM else "DDIM"
    save_path = args.save_path+"/%s/%d_steps/result_%s_steps_%d_ratio_%f_schedule_%s" % (
        model_id, num_inference_steps, model_name, num_inference_steps, ratio, schedule)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load model and scheduler
    from diffusers import DDIMPipeline
    ddim = DDIMPipeline.from_pretrained(model_id).to("cuda")
    ddim.scheduler.set_config_res(to_RDDM, ratio, schedule, adjust_alpha)

    # gen img
    total_n_samples = 50000
    img_id = len(glob.glob(f"{save_path}/*"))
    n_rounds = (total_n_samples - img_id) // batch_size+1
    if img_id!=0:
        set_seed(img_id)
    for i in range(n_rounds):
        # run pipeline in inference (sample random noise and denoise)
        image = ddim(num_inference_steps=num_inference_steps, batch_size=batch_size).images


        for k in range(batch_size):
            file_name = f'{save_path}/{img_id}.png'
            if img_id>=total_n_samples:
                break
            image[k].save(file_name)
            img_id += 1

        print("save img:"+file_name)


if __name__ == '__main__':
    main()
