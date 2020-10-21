# import json
# def get_label_num(tags_file):
#     labels = []
#     with open(tags_file,'r',encoding='utf-8') as f:
#         for label in f.readlines():
#             labels.append(label.strip())
#     return labels,len(labels) #返回标签列表，标签个数
# labels,num = get_label_num('./data/tags.txt')
#
#
# def _read_tsv(cls, input_file, quotechar=None):
#     """Reads a tab separated value file."""
#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = []
#         line = f.readline()
#         count = 0
#         while line:
#             if count == 5:
#                 break
#             count += 1
#             line = json.loads(line)
#             for text in line:
#                 temp = []
#                 labels = str(','.join(text['labels']))
#                 sentence = str(text['sentence'])
#                 # if labels == '':
#                 #   labels = 'DV0'
#                 temp.append(labels)
#                 temp.append(sentence)
#                 lines.append(temp)
#             line = f.readline()
#     return lines
#
# lines = _read_tsv('./data/input.json',"./data/input.json")
# print(labels,num)
# print(lines)
# for (i, line) in enumerate(lines):
#     print(i,line)
# import random
# a = [1,2,3,4,5,6,7,8,9]
# random.shuffle(a)
# print(a)

# import time
# start = time.clock()
# # 当中是你的程序
# end = time.clock()
# elapsed = (time.clock() - start)
# print("Time used:", elapsed)

# data = './output/result.txt'
# lines = open(data,'r',encoding='utf-8')
# def count_max_f(file):
#     lines = open(file,'r',encoding='utf-8').readlines()
#     max_f = 0
#     line_count = 0
#     max_line_count = 0
#     k = -12
#
#     for line in lines:
#         # if 'macro avg' in line:
#         #     print(line)
#         #     print(line.split('\t')[0].split('    ')[1].strip())
#         #     print(line.split('\t')[0].split('    ')[2].strip())
#         #     print(line.split('\t')[0].split('    ')[3].strip())
#         line_count += 1
#         if 'macro avg' in line:
#             f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(line.split('\t')[0].split('    ')[2].strip()) / (float(line.split('\t')[0].split('    ')[1].strip()) + float(line.split('\t')[0].split('    ')[2].strip()))
#             if(f1 > max_f):
#                 max_f = f1
#                 max_line_count = line_count
#
#     for i in range(13):
#         print(lines[max_line_count + k])
#         k += 1
#
#     print('The true max f1:',max_f)
#
# count_max_f(data)

# text = '啊的卡上科技大厦'
# print(text[10:20])
#
# file_road = './data/__Result_nolap.txt'
# lines = open(file_road,'r',encoding='utf-8').readlines()
# negative = 0
# positive = 0
# all = 0
# for line in lines:
#     if line.split(' ')[3].strip() == 'Negative':
#         negative += 1
#     else:
#         positive += 1
#     all += 1
#
# print(negative,positive,all,negative + positive)
# a = ['1','2','3','4','5']
# print(' '.join(a))
# a = '11111111'
# a[2:4] = '22'
# print(a)
import numpy as np
# embedding_file = 'chinese_L-12_H-768_A-12/bert_embed.txt.npy'
# lines = np.load(embedding_file, encoding="latin1")
# print("----type----")
# print(type(lines))
# print("----shape----")
# print(lines.shape)
# print("----data----")
# print(lines)
#
# arr = np.arange(3)
# arr = np.expand_dims(np.expand_dims(arr, 1), axis=2)
# print(np.tile(arr, [1, 1, 2]))

# import tensorflow as tf

# #定义变量a
# a=tf.Variable([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
#
# #定义索引
# indics=[[0,0,0],[0,1,1],[0,1,2]]
#
# #把a中索引为indics的值取出
# b=tf.gather_nd(a,indics)
#
# #初始化
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     #执行初始化
#     sess.run(init)
#
#     #打印结果
#     print(a.eval())
#     print(b.eval())

import tensorflow as tf
import numpy as np

# indices = [2] * 32
# range_ = np.arange(32)
# print(range_, indices)
# indices = np.stack([range_, indices], 1)
# indices_1 = [3] * 32
# indices_1 = np.stack([range_, indices_1], 1)
# indices = np.stack([indices, indices_1], 1)
# print(indices)
# print('66666')
# all_features = '我 是 一 个 句 子'
# word_index = ['*']*200
# # if len(word_index) < 100:
# #     for add_word in range(0,(100 - len(word_index))):
# #         word_index.append(str(len(all_features.split(' '))))
# if len(word_index) > 100:
#     word_index = word_index[0:100]
# print(len(word_index))
a = 'ABCD'
print(a.lower())
