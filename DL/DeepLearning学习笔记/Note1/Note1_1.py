from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("训练图片个数与尺寸： ", train_images.shape, "标签数： ", len(train_labels))
print("测试图片数量与尺寸： ", test_images.shape, "标签数： ", len(test_labels))
# 网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))
# 编译
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 数据预处理,将其变换为网络要求的形状，并缩放到所有值都在 [0, 1] 区间
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 对标签进行分类编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 训练模型，epochs表示训练遍数，batch_size表示每次喂给网络的数据数目
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 检测在测试集上的正确率
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('正确率: ', test_acc)