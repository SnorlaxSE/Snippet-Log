# 单标签、多分类问题: 新闻主题分类
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
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


# 将数据限定为前10000个最常出现的单词
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000, path="/Users/Captain/dataset/reuters/reuters.npz")
# 新闻解析
word_index = reuters.get_word_index(path="/Users/Captain/dataset/reuters/reuters_word_index.json")
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
# 索引减去了3，因为 0、1、2 是为“padding”( 填 充 )、“start of
# sequence”(序列开始)、“unknown”(未知词)分别保留的索引
decoded_newswire = ' '.join([reversed_word_index.get(i-3, '?') for i in train_data[0]])
print(decoded_newswire)
# 标签的索引范围为0 - 45
print(np.amax(train_labels))

# 数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 留出1000验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()