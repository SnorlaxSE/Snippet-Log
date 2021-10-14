# 可视化卷积神经网络的过滤器
from keras import backend as K
import numpy as np
from keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.preprocessing import image


# 将张量转换为有效图像
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    # 将 x 裁切（clip）到 [0, 1] 区间
    x = np.clip(x, 0, 1)
    x *= 255
    # 将 x 转换为 RGB 数组
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 生成过滤器可视化
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for _ in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


# 为过滤器的可视化定义损失张量
model = VGG16(weights='imagenet', include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])
# 　获取损失相对于输入的梯度
grads = K.gradients(loss, model.input)[0]
# 将梯度张量除以其 L2 范数来标准化
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# 　给定 Numpy 输入值，得到 Numpy 输出值
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 　通过随机梯度下降让损失最大化
# 从一个带噪声的随机图像开始
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
plt.imshow(image.array_to_img(input_img_data[0]))
plt.show()
step = 1.
for i in range(40):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step


def draw_layer_filter(layer_name):
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    results = image.array_to_img(results)
    plt.imshow(results)
    plt.show()


draw_layer_filter(layer_name='block1_conv1')
draw_layer_filter(layer_name='block4_conv1')