%% Multi-Stage Progressive Image Restoration
%% Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
%% https://arxiv.org/abs/2102.02808

close all;clear all;

% datasets = {'GoPro'};
datasets = {'GoPro'};
num_set = length(datasets);

tic
delete(gcp('nocreate'))
% 检查核心数量
ncores=feature('numCores');
disp([num2str(ncores) ' cores found'])
parpool('local',ncores);

for idx_set = 1:num_set
    file_path = '/home/backup/program_results/diffusion/去模糊/res_diffusion_tempalte_celeba_pred_res_noise_deblurring_gopro_resize_256_two_op_timestep2/';
    gt_path = '/home/backup/program_results/diffusion/dataset/gopro/test/target/';
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        parfor j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           %%%%
           [M,N,C] = size(input);
           gt=imresize(gt,[M,N]);
           %%%%
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
           disp(num2str(idx_set)+"-dataset:"+num2str(j));
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    
    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

end
delete(gcp('nocreate'))
toc
