# Calculate lpips

1. Install [lpips](https://github.com/richzhang/PerceptualSimilarity). 
   1. `pip install lpips`.
2. Calculating `lpips` requires that two images have the same resolution. If the resolutions are different, you can modify the resolution and save the new image. 
   1. `python resize_256_and_save.py load_folder save_folder`. 
   2. `load_folder` is the path for the original image and the `save_folder` is the save path for the modified resolution image.
3. Calculate lpips.
   1.  `python lpips_2dirs.py -d0 neywork_out_256 -d1 test_256`. 
   2.  `neywork_out_256` is the path of the network output folder, while `test_256` is the path of the ground truth folder.