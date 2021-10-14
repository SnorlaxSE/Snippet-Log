# one-hot编码Demo
from keras.preprocessing.text import Tokenizer


samples = ['The cat sat on the mat.', 'The dog ate my homework.']
# 只考虑前10个最常见的单词
tokenizer = Tokenizer(num_words=10)
# 构建单词索引
tokenizer.fit_on_texts(samples)
# 找回单词索引
word_index = tokenizer.word_index
print(word_index)
# 将字符串转换为整数索引组成的列表
sequences = tokenizer.texts_to_sequences(samples)
print("转换成的索引序列 ", sequences)
text = tokenizer.sequences_to_texts(sequences)
print("转会的文本 ", text)
# 得到 one-hot 二进制表示
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
one_num = 0
for items in one_hot_results:
    for item in items:
        if item == 1:
            one_num += 1
print("1的数量为 ", one_num)
print(one_hot_results)

'''
Using TensorFlow backend.
{'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'ate': 7, 'my': 8, 'homework': 9}
转换成的索引序列  [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
转会的文本  ['the cat sat on the mat', 'the dog ate my homework']
1的数量为  10
[[0. 1. 1. 1. 1. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 1. 1. 1. 1.]]
 '''