# Residual Denoising Diffusion Models

This repository is the official implementation of Residual Denoising Diffusion Models.


Note:
1. The current setting is to train two unets (one to estimate the residuals and one to estimate the noise), which can be used to explore partially path-independent generation process.
2. Other tasks can modify the corresponding experimental settings. See Table 4 in the Appendix.
3. The code is being updated.

## Requirements

To install requirements:

```
conda env create -f install.yaml
```

## Training

To train RDDM, run this command:

```train
python train.py
```
or
```train
accelerate launch train.py
```

## Evaluation

To evaluate image generation, run:

```eval
cd eval/image_generation_eval/
python fid_and_inception_score.py path_of_gen_img
```

For image restoration, MATLAB evaluation codes in `./eval`.

## Pre-trained Models

The pre-trained models will be provided later.

## Results

See Table 2 in main paper.

## Citation
If you find our work useful in your research, please consider citing:
```
@article{liu2023residual,
    title={Residual Denoising Diffusion Models}, 
    author={Jiawei Liu and Qiang Wang and Huijie Fan and Yinong Wang and Yandong Tang and Liangqiong Qu},
    year={2023},
    journal={arXiv preprint arxiv:2308.13712}
}
```
## Contact
Please contact Jiawei Liu if there is any question (liujiawei18@mails.ucas.ac.cn).