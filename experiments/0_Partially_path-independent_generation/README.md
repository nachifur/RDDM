# partially path-independent generation process

The current setting is to train two unets (one to estimate the residuals and one to estimate the noise), which can be used to explore partially path-independent generation process.

**1. Load Pre-trained Models**

Download [the pre-trained models (two unets, deresidual+denoising)](https://rec.ustc.edu.cn/share/3d8d9200-4e7e-11ef-b0ee-250e7e41f368) for [partially path-independent generation process](https://github.com/nachifur/RDDM/tree/main/experiments/0_Partially_path-independent_generation).
```
cd ./RDDM/experiments/0_Partially_path-independent_generation
cp model-100.pt ./results/sample/
```
[Then, you should set the path of celeba dataset.](https://github.com/nachifur/RDDM/blob/d5a6b82de5166b92f22570e258f9e590e23231ff/experiments/0_Partially_path-independent_generation/train.py#L59)
```
python test.py
```


**2. Differences in code compared to other tasks:**
Other tasks need to modify 

a) `[self.alphas_cumsum[t]*self.num_timesteps, self.betas_cumsum[t]*self.num_timesteps]]` -> `[t,t]` (in [L852](https://github.com/nachifur/RDDM/blob/50d7dc3670a68dfe89c411a9445cc824b4fcd911/src/residual_denoising_diffusion_pytorch.py#L852) and [L1292](https://github.com/nachifur/RDDM/blob/50d7dc3670a68dfe89c411a9445cc824b4fcd911/src/residual_denoising_diffusion_pytorch.py#L1292)).

b) For image restoration, `generation=False` in [L120](https://github.com/nachifur/RDDM/blob/ee4df22b672772a46b48251b0f56d82489d6adf0/train.py#L120), `convert_to_ddim=False` in [L640](https://github.com/nachifur/RDDM/blob/46ffd50f858a59fc3b43e538d501d991af3c1472/src/residual_denoising_diffusion_pytorch.py#L640)  and [L726](https://github.com/nachifur/RDDM/blob/11d06d5f389e3953fdedcf33fffbb81ff4d1583a/src/residual_denoising_diffusion_pytorch.py#L726C9-L726C31). 

c) uncomment [L726](https://github.com/nachifur/RDDM/blob/7cbb21139f33e7fe453aa1e8105a2371fa8eb5ee/src/residual_denoising_diffusion_pytorch.py#L1119) for simultaneous removal of residuals and noise. 

d) **modify the corresponding experimental settings (see Table 4 in the Appendix)**.
