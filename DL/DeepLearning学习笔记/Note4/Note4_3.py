# 可视化类激活的热力图
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image
import cv2

model = VGG16(weights='imagenet')
img_path = 'testImg.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
# 添加一个维度，将数组转换为(1, 224, 224, 3) 形状的批量
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])
# 索引编号
np.argmax(preds[0])
# 来使用 Grad-CAM 算法展示图像中哪些部分最像非洲象
african_elephant_output = model.output[:, 386]
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

# 热力图后处理
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# 将热力图与原始图像叠加
img = Image.open(img_path).convert('RGB')
img = np.array(img)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img
cv2.imwrite('elephant_cam.jpg', superimposed_img)

superimposed_img = image.array_to_img(superimposed_img)
plt.imshow(superimposed_img)
plt.show()

'''
Using TensorFlow backend.
2019-01-15 14:26:17.035252: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiledto use: AVX2 FMA
Predicted: [('n02504458', 'African_elephant', 0.87405443), ('n01871265', 'tusker', 0.11621151), ('n02504013', 'Indian_elephant', 0.008261557)]
'''