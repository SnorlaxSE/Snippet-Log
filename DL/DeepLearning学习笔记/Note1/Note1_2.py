# 根据电影评论的文字内容将其划分为正面或负面
from keras.datasets import imdb
import os
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


# 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # results[i] 的指定索引设为 1
        results[i, sequence] = 1
    return results


data_url_base = "/Users/Captain/dataset"
# 下载数据且只保留出现频率最高的前10000个单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000, path=os.path.join(data_url_base, "imdb.npz"))

# 将某条评论迅速解码为英文单词
# word_index 是一个将单词映射为整数索引的字典
word_index = imdb.get_word_index(path=os.path.join(data_url_base, "imdb_word_index.json"))
# 将整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 索引减去了 3,因为 0、1、2是为“padding”(填充)、
# “start of sequence”(序列开始)、“unknown”(未知词)分别保留的索引
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# 将数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 设计网络
# 两个中间层,每层都有 16 个隐藏单元
# 第三层输出一个标量,预测当前评论的情感
# 中间层使用 relu 作为激活函数,最后一层使用 sigmoid 激活以输出一个 0~1 范围内的概率值
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 模型编译
# binary_crossentropy二元交叉熵
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
# 得到训练过程中的所有数据
history_dict = history.history
print(history_dict.keys())

# 绘制训练损失和验证损失
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')   # 'bo' 蓝色圆点
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') # 'b' 蓝色实线
plt.title("Training and Validation loss")
plt.xlabel('Epochs')
plt.legend()
plt.show()

# 绘制训练精度和验证精度
# plt.clf() 清空图像
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()