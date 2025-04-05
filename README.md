# Residual Denoising Diffusion Models

[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Residual_Denoising_Diffusion_Models_CVPR_2024_paper.html)|[arxiv](https://arxiv.org/abs/2308.13712)|[youtube](https://www.youtube.com/watch?v=E-ObZs32fEU)|[blog](https://twitter.com/nachifur/status/1762730191707881537)|[中文论文(ao9l)](https://rec.ustc.edu.cn/share/60cb4770-1b6a-11ef-8e9e-332aeb6c199a)|[中文视频](https://cmdr.com.cn/lectureHall/lectureRoomDetail?liveUid=58e63bb51116d7c01f37dfee1354b043)|[中文博客](https://www.zhihu.com/question/645935461/answer/3410873004)

This repository is the official implementation of Residual Denoising Diffusion Models.

<p align="center">
<a href="https://cvpr.thecvf.com/virtual/2024/poster/31373" target="_blank">
<img width="800" height="400" img align="center" alt="RDDM" src="https://github.com/nachifur/RDDM/blob/main/poster/Jiawei_9969.png" />
</a>
</p>

## Requirements

To install requirements: ([If an error occurs, you may need to install the packages one by one](https://github.com/nachifur/RDDM/issues/41#issuecomment-2477808693).)

```
conda env create -f install.yaml
```


## Dataset

[Raindrop](https://github.com/rui1996/DeRaindrop) ([test-a for test](https://github.com/rui1996/DeRaindrop))

[GoPro](https://github.com/swz30/MPRNet/blob/main/Deblurring/Datasets/README.md)

[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)

SID-RGB: [kexu](https://kkbless.github.io/) or [download](https://drive.google.com/drive/folders/1-psXDjeW4FiRdLjc9idABsxGPo1Kn1jR)

[LOL](https://daooshee.github.io/BMVC2018website/)

[CelebA](https://github.com/nachifur/RDDM/issues/8#issuecomment-1978889073)

## Training

To train RDDM, run this command:

```train
cd experiments/xxxx
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

[The pre-trained models (two unets, deresidual+denoising)](https://rec.ustc.edu.cn/share/3d8d9200-4e7e-11ef-b0ee-250e7e41f368) for [partially path-independent generation process](https://github.com/nachifur/RDDM/tree/main/experiments/0_Partially_path-independent_generation).

## Results

See Table 3 in main paper.

**For image restoration:**

[Raindrop](https://rec.ustc.edu.cn/share/c20ea640-4e7e-11ef-b29e-b1b12149494a)

[GoPro](https://rec.ustc.edu.cn/share/f9deffc0-4e7e-11ef-b4dd-b51790f24839)

[ISTD](https://rec.ustc.edu.cn/share/da867b10-4e7e-11ef-b21d-b3131e611f52)

[LOL](https://rec.ustc.edu.cn/share/e9c00ab0-4e7e-11ef-89a0-292c4c37c153)

[SID-RGB](https://rec.ustc.edu.cn/share/b213c330-4e7e-11ef-9b3e-957f50ca7d9b)


**For image generation (on the CelebA dataset):**

We can convert a pre-trained DDIM to RDDM by coefficient transformation (see [1_Image_Generation_convert_pretrained_DDIM_to_RDDM](https://github.com/nachifur/RDDM/tree/main/experiments/1_Image_Generation_convert_pretrained_DDIM_to_RDDM)).

## Experiments 
https://github.com/nachifur/RDDM/tree/main/experiments

[0_**Partially_path-independent**_generation](https://github.com/nachifur/RDDM/tree/main/experiments/0_Partially_path-independent_generation)

[1_Image_Generation_convert_pretrained_**DDIM_to_RDDM**](1_Image_Generation_convert_pretrained_DDIM_to_RDDM)

[2_**Image_Restoration**_deraing_raindrop_noise1](https://github.com/nachifur/RDDM/tree/main/experiments/2_Image_Restoration_deraing_raindrop_noise1)

[3_**Automatic-Objective-Selection-Algorithm**_test_lol_timestep5_1e-2_reinit_weight](https://github.com/nachifur/RDDM/tree/main/experiments/3_AOSA_test_lol_timestep5_1e-2_reinit_weight)

[4_**Image_Inpainting**_imgsize64_batch64_pred_res_noise_centermask_wo_mask_wo_input](https://github.com/nachifur/RDDM/tree/main/experiments/4_Image_Inpainting_imgsize64_batch64_pred_res_noise_centermask_wo_mask_wo_input)

[5_**Image_translation**_dog_to_cat_wo_input_imgsize64_batch64_pred_res_noise](https://github.com/nachifur/RDDM/tree/main/experiments/5_Image_translation_dog_to_cat_wo_input_imgsize64_batch64_pred_res_noise)

[6_**Image_Generation**_table2_decreased_alpha_increased_beta](https://github.com/nachifur/RDDM/tree/main/experiments/6_Image_Generation_table2_decreased_alpha_increased_beta)

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Jiawei and Wang, Qiang and Fan, Huijie and Wang, Yinong and Tang, Yandong and Qu, Liangqiong},
    title     = {Residual Denoising Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {2773-2783}
}
```
## Contact
Please contact Jiawei Liu (liujiawei18@mails.ucas.ac.cn) or Liangqiong Qu (https://liangqiong.github.io/) if there is any question.
