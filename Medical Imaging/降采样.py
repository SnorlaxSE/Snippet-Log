import scipy.ndimage
import SimpleITK as sitk
import numpy as np


def pet_interpolation(target_shape, data_arr):

    data_arr_shape = data_arr.shape
    print(data_arr_shape)
    
    x, y, z = target_shape[0], target_shape[1], target_shape[2]
    new_data_arr = scipy.ndimage.zoom(data_arr, (x/data_arr.shape[0], y/data_arr.shape[1], z/data_arr.shape[2]), order=3)
    print(new_data_arr.shape)

    return new_data_arr



if __name__ == "__main__":

    file = 'CN_71_009_S_0751_20061023_UR_N3_non_wm.nii'
    output_nii = 'CN_71_009_S_0751_20061023_UR_N3_non_wm_resample.nii'
    output_npy = 'CN_71_009_S_0751_20061023_UR_N3_non_wm_resample.npy'

    image = sitk.ReadImage(file)
    data_arr = sitk.GetArrayFromImage(image)
    target_shape = (160, 160, 96)
    new_data_arr = pet_interpolation(target_shape, data_arr)

    # save as nii file
    image = sitk.GetImageFromArray(new_data_arr)
    sitk.WriteImage(image, output_nii)

    # # save as npy file
    # np.save(file=output_npy, arr=new_data_arr)
    pass
