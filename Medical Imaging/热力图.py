from keras.models import load_model
import numpy as np
from keras import backend as K
import pylab
import os
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import pdb
import imageio
import os
import cv2
import scipy.ndimage

def normalize(data_arr):
    max_ = np.max(data_arr)
    min_ = np.min(data_arr)
    data_arr = (data_arr - min_) / (max_ - min_)
    return data_arr

def read_nii(nii_file):
    nii_image = sitk.ReadImage(nii_file)
    nii_arr = sitk.GetArrayFromImage(nii_image)
    return nii_arr

def sensitivity_analysis(img_path, model_path, model_type, relu=True, classes=2):
    # grad-cam
    nii_arr = read_nii(img_path)
    nii_arr = nii_arr[np.newaxis, :, :, :, np.newaxis]
    
    model = load_model(model_path)
    model.summary()

    pred = model.predict(nii_arr)

    pred_true = 0
    if classes == 2:
        predict_score = np.squeeze(pred)
        if predict_score >= 0.5:
            pred_true = 1
    else:
        predict = np.squeeze(pred)
        pred_true = int(np.where(predict == np.max(predict))[0])
        
    print(pred)

    print("pred_true: ", pred_true)
    pre_output = model.output[:, pred_true]   # !!!
    # import tensorflow as tf
    # session = tf.Session()
    # array = session.run(pre_output)
    # print(array)
    # with tf.Session() as sess:
    #     print(model.output.eval())
    #     print(pre_output.eval())

    if model_type == "conv":
        # 最后一层卷积层名字
        # last_conv_layer = model.get_layer('batch_normalization_4')

        # # 尝试第一层卷积层 效果好
        last_conv_layer = model.get_layer('conv3d_{}'.format(conv_order))

        

    grads = K.gradients(pre_output, last_conv_layer.output)[0]
    # pdb.set_trace()

    pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))  # data dim
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([nii_arr])
    # print("pooled_grads_value.shape: ", pooled_grads_value.shape, np.max(pooled_grads_value), np.min(pooled_grads_value))
    # print("conv_layer_output_value.shape: ", conv_layer_output_value.shape, np.max(conv_layer_output_value), np.min(conv_layer_output_value))

    if relu:  # ...
        conv_layer_output_value[np.where(conv_layer_output_value < 0)] = 0

    # print("pooled_grads_value.shape: ", pooled_grads_value.shape, np.max(pooled_grads_value), np.min(pooled_grads_value))
    # print("conv_layer_output_value.shape: ", conv_layer_output_value.shape, np.max(conv_layer_output_value), np.min(conv_layer_output_value))

    conv_layer_output_value = normalize(conv_layer_output_value)
    pooled_grads_value = normalize(pooled_grads_value)

    # print("pooled_grads_value.shape: ", pooled_grads_value.shape, np.max(pooled_grads_value), np.min(pooled_grads_value))
    # print("conv_layer_output_value.shape: ", conv_layer_output_value.shape, np.max(conv_layer_output_value), np.min(conv_layer_output_value))

    layer_number = len(pooled_grads_value) 
    for i in range(layer_number):
        conv_layer_output_value[:,:,:,i] *= pooled_grads_value[i]

    # along the last dim calculate the mean value
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # remove the value which less than 0
    heatmap = np.maximum(heatmap, 0)
    # uniformization
    min_ = np.min(heatmap)
    max_ = np.max(heatmap)
    heatmap = (heatmap - min_) / (max_ - min_)

    return heatmap

def png2gif(png_dir, gif_path, slice_num):
    imgs = []
    file_names = []
    for i in range(slice_num):
        filepath = os.path.join(png_dir, str(i) + '.png')
        file_names.append(filepath)
    for filepath in file_names:
        image = imageio.imread(filepath)  # (80, 80, 4)
        global target_width_height
        from skimage.transform import resize
        image = (resize(image, (target_width_height, target_width_height, 4))*255).astype(np.uint8)
        imgs.append(image)
    imageio.mimsave(gif_path, imgs, duration=0.3)
    print('{}  done.'.format(gif_path))


def draw(heatmap, data, png_dir):
    
    plt.axis('off')
    for i in range(len(heatmap)):
        data_slice = data[i, :, :]
        heatmap_slice = heatmap[i, :, :]
        width, height = data_slice.shape[0], data_slice.shape[1]

        plt.imshow(data_slice, cmap='bone')
        plt.imshow(heatmap_slice, cmap='rainbow', alpha=0.3)

        plt.gcf().set_size_inches(width / 100, height / 100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path = os.path.join(png_dir, str(i) + '.png')
        plt.savefig(save_path, bbox_inched='tight')

        print("{} done.".format(png_dir))


if __name__ == '__main__':

    model_path = r'HCI_WM_PET_UR_ADvsMCIvsCN_fold_dataset_halve_f4_conv_8_16_32_64.acc.b16.h5'
    img_path = r'AD_60_033_S_1283_20070313_UR_N3_wm_UR_halve_1.nii.gz'
    conv_order = "4"

    nii_arr = read_nii(img_path)
    nii_arr = normalize(nii_arr)
    heatmap = sensitivity_analysis(img_path=img_path, model_path=model_path, model_type="conv", classes=3)

    from skimage.transform import resize
    heatmap = resize(heatmap, (48, 80, 80))
    png_dir = "occlusion_conv_{}".format(conv_order)
    target_width_height = 512
    gif_path = os.path.join(png_dir, "conv_{}_{}.gif".format(conv_order, str(target_width_height)))
    
    draw(heatmap, nii_arr, png_dir)
    
    png2gif(png_dir, gif_path, slice_num=len(heatmap))