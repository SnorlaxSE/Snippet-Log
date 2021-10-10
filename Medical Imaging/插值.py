import scipy.ndimage
import SimpleITK as sitk

def pet_interpolation(target_shape, data, nii_file, npy_file):

    data_shape = data.shape
    print(data_shape)
    
    x, y, z = target_shape[0], target_shape[1], target_shape[2]
    new_data_arr = scipy.ndimage.zoom(data, (x/data.shape[0], y/data.shape[1], z/data.shape[2]), order=3)
    print(new_data_arr.shape)

    # save as nii file
    image = sitk.GetImageFromArray(new_data_arr)
    sitk.WriteImage(image, nii_file)

    # save as npy file
    np.save(file=npy_file, arr=new_data_arr)