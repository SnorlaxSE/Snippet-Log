import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SimpleITK as sitk
import numpy as np
from scipy import stats
import shutil
import os


def get_img_mode(img_arr, is_show=False):

    # arr_rm_0
    img_arr = img_arr.astype(np.int) # necessary

    flatten_arr = img_arr.flatten()
    remove_ixs=np.where(flatten_arr==0)[0]
    flatten_arr_rm_0=np.delete(flatten_arr,remove_ixs)
    global_max = np.max(img_arr)
    global_mode = stats.mode(flatten_arr_rm_0)[0][0]
    # print("global max:", global_max)
    # print("global mode:", global_mode)
    
    # slice
    slice_num = int(img_arr.shape[1] / 2)
    # print("slice_num:", slice_num)
    image_slice_array = img_arr[:,slice_num,:]

    # slice_rm_low_threshold (global_max*0.15) # 去除噪声对计算mode的影响
    flatten_slice_arr = image_slice_array.flatten()
    slice_remove_ixs=np.where(flatten_slice_arr< global_max*0.15)[0]
    flatten_slice_arr_rm_0=np.delete(flatten_slice_arr,slice_remove_ixs)
    slice_max = np.max(flatten_slice_arr_rm_0)
    slice_mode = stats.mode(flatten_slice_arr_rm_0)[0][0]
    # print("local max:", slice_max)
    # print("local mode:", slice_mode)

    # print("local_mode/global_max:", slice_mode/global_max)
    print("global_max/local_mode:", global_max/slice_mode)

    high_threshold = int(slice_mode * 2.2)

    print("global_max:{} high_threshold:{} slice_mode:{}".format(global_max, high_threshold, slice_mode))


    # show
    if is_show:

        plt.subplot(2, 1, 1)
        plt.imshow(image_slice_array, cmap=cm.gray)
        plt.axis("off")
        plt.subplot(2, 1, 2)
        plt.hist(flatten_slice_arr_rm_0, bins=100) # flatten可以将矩阵转化成一维序列
        plt.show()


    return slice_mode, global_max



if __name__ == "__main__":

    src_dir = './HCI_AD_MRI_N3_dataset_Age_skull_MNI'
    out_dir = './HCI_AD_MRI_N3_dataset_Age_skull_MNI_HT'

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if 'nii' in file:
                filename_prefix = file.split('.')[0]
                img_path = os.path.join(root, file)
                out_name = filename_prefix + "_ht_norm.nii.gz"
                out_path = os.path.join(out_dir, out_name)
                
                # image
                image = sitk.ReadImage(img_path)
                img_arr = sitk.GetArrayFromImage(image)
                print(img_arr.shape) # (166, 256, 256)

                slice_mode, global_max = get_img_mode(img_arr, is_show=False)
                # print(slice_mode, global_max)

                # threshold cut off
                high_threshold = int(slice_mode * 2.2)

                if global_max > high_threshold:
                    img_arr[img_arr > high_threshold] = high_threshold
                    print(np.max(img_arr))


                # 最大最小归一化
                norm_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

                threshold_norm_img = sitk.GetImageFromArray(norm_arr)
                sitk.WriteImage(threshold_norm_img, out_path)
                print("{} done.".format(out_path))
