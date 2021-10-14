# 使用数据增强的方法增加数据
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_path = '/Users/Captain/dataset/dogVScat/test/dogs/dog.77.jpg'
# 加载图片并调整尺寸
img = np.asarray(image.load_img(img_path, target_size=(150, 150)))
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
plt.imshow(img)
plt.title('original img')
plt.show()


img = img.reshape((1, ) + img.shape)
i = 0
for item in datagen.flow(img, batch_size=1):
    item = image.array_to_img(item[0])
    plt.subplot(2, 2, i+1)
    plt.imshow(item)
    i += 1
    plt.title('generated img ' + str(i))
    if i % 4 == 0:
        break
plt.show()