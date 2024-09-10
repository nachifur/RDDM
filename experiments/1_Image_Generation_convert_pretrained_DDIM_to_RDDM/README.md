
# Image Generation (convert a pre-trained DDIM to RDDM by coefficient transformation)
We use a pre-trained DDIM by `from diffusers import DDIMPipeline` (the version of `diffusers` is 0.15.1).  See [huggingface.co/google/ddpm-celebahq-256](https://huggingface.co/google/ddpm-celebahq-256) for more details.

1. find `scheduling_ddim.py`, e.g., `/home/liu/anaconda3/envs/diffusion/lib/python3.7/site-packages/diffusers/schedulers/scheduling_ddim.py`.
2. `cp /home/liu/anaconda3/envs/diffusion/lib/python3.7/site-packages/diffusers/schedulers/scheduling_ddim.py ./src/scheduling_ddim.py`
3. Copy the modified file (`scheduling_ddim_our.py`) to the original file (`diffusers/schedulers/scheduling_ddim.py`).
```
cp ./src/scheduling_ddim_our.py /home/liu/anaconda3/envs/diffusion/lib/python3.7/site-packages/diffusers/schedulers/scheduling_ddim.py
```
4. DDIM: sample img by 
```
python test.py --save_path /home/xxx/xxx --num_inference_steps 10 --to_RDDM 0 --ratio  -1 --gpu_id 0
```
5. DDIM_to_RDDM: sample img by 
```
python test.py --save_path /home/xxx/xxx --num_inference_steps 10 --to_RDDM 1 --ratio  -1 --gpu_id 0
```
6. For `Analysis of readjusting coefficient schedules`, you can get the result of `P(1-x,0.3)` in (g).
```
python test.py --save_path /home/liu/disk12t/liu_data/diffusion/res_ddim_same_weight --num_inference_steps 100 --to_RDDM 1 --ratio 0.3 --gpu_id 0 --adjust_alpha 1 --schedule decreased
```
7. `cp ./src/scheduling_ddim.py /home/liu/anaconda3/envs/diffusion/lib/python3.7/site-packages/diffusers/schedulers/scheduling_ddim.py`
