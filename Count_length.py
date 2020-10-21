# -*- coding:utf-8 -*-
"""
@project:relaiton_extraction
@author:yangwz
@time:2019/10/16-8:49
"""

import matplotlib.pyplot as plt
import numpy as np

file_path = './data/__Result_nolap.txt'
lines = open(file_path,'r',encoding='utf-8').readlines()
print(len(lines))
sentence_lengths = dict()
all_lengths = dict()
for line in lines:
    sentence = line.split(' ')[2]
    e1_e2_sentence = line.split(' ')[0] + line.split(' ')[1] + line.split(' ')[2]
    if len(sentence) not in sentence_lengths:
        sentence_lengths[len(sentence)] = 1
    else:
        sentence_lengths[len(sentence)] += 1

    if len(e1_e2_sentence) not in all_lengths:
        all_lengths[len(e1_e2_sentence)] = 1
    else:
        all_lengths[len(e1_e2_sentence)] += 1

print("Sentence_length:",sentence_lengths)
print("All_lengths:",all_lengths)

print("Sentence_length:",sorted(sentence_lengths.items(),key = lambda x:x[1],reverse = True))
print("All_lengths:",sorted(all_lengths.items(),key = lambda x:x[1],reverse = True))

sentence_lengths = sorted(sentence_lengths.items(),key = lambda x:x[0],reverse = True)
all_lengths = sorted(all_lengths.items(),key = lambda x:x[0],reverse = True)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
x_data = []
y_data = []

for each in sentence_lengths:
    x_data.append(each[0])
    y_data.append(each[1])

# 设置标题
plt.title("ACE 2005 数据分布折线图")
# 为两条坐标轴设置名称
plt.xlabel("句子长度")
plt.ylabel("数量")

# 绘折线图
plt.plot(x_data,y_data,c = 'k')
plt.legend()
plt.show()

# 绘柱状图
# plt.bar(x=x_data, height=y_data, label='Sentence_Length', color='steelblue', alpha=0.8)
plt.bar(x=x_data, height=y_data, color='steelblue', alpha=0.8)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for x, y in enumerate(y_data):
#     plt.text(x, y, '%s' % y, ha='center', va='bottom')

# 设置标题
plt.title("ACE 2005 数据分布柱状图")
# 为两条坐标轴设置名称
plt.xlabel("句子长度")
plt.ylabel("数量")
# 显示图例
plt.legend()
plt.show()

print(x_data)
print(y_data)

# # 构建数据
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# x_data = ['2012', '2013', '2014', '2015', '2016', '2017', '2018']
# y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
# y_data2 = [52000, 54200, 51500,58300, 56800, 59500, 62700]
# # 绘图
# plt.bar(x=x_data, height=y_data, label='C语言基础', color='steelblue', alpha=0.8)
# plt.bar(x=x_data, height=y_data2, label='Java基础', color='indianred', alpha=0.8)
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for x, y in enumerate(y_data):
#     plt.text(x, y + 100, '%s' % y, ha='center', va='bottom')
# for x, y in enumerate(y_data2):
#     plt.text(x, y + 100, '%s' % y, ha='center', va='top')
# # 设置标题
# plt.title("Java与Android图书对比")
# # 为两条坐标轴设置名称
# plt.xlabel("年份")
# plt.ylabel("销量")
# # 显示图例
# plt.legend()
# plt.show()

# all_sum = 0
# print("训练数据分布：")
# file_path = './data/train.txt'
# data = dict()
# lines = open(file_path,'r',encoding='utf-8').readlines()
# for line in lines:
#     if line.split(' ')[3].strip() not in data:
#         data[line.split(' ')[3].strip()] = 1
#     else:
#         data[line.split(' ')[3].strip()] += 1
# print("训练数据条数：",len(lines))
# print(data)
# for k,v in data.items():
#     all_sum += v
# print(all_sum)
#
# for k,v in data.items():
#     if "Negative" == k:
#         continue
#     x_data.append(k)
#     y_data.append(v)
# # 绘柱状图
# # plt.bar(x=x_data, height=y_data, label='Sentence_Length', color='steelblue', alpha=0.8)
# plt.bar(x=x_data, height=y_data, color='steelblue', alpha=0.8,width=0.5)
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # for x, y in enumerate(y_data):
# #     plt.text(x, y, '%s' % y, ha='center', va='bottom')
#
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for x, y in enumerate(y_data):
#     plt.text(x, y, '%s' % y, ha='center', va='bottom')
# # 设置标题
# plt.title("ACE 2005 数据分布柱状图")
# # 为两条坐标轴设置名称
# plt.xlabel("关系类别")
# plt.ylabel("数量")
# # 显示图例
# plt.legend()
# plt.show()
#
# all_sum = 0
# print("测试数据分布")
# file_path = './data/test.txt'
# data = dict()
# lines = open(file_path,'r',encoding='utf-8').readlines()
# for line in lines:
#     if line.split(' ')[3].strip() not in data:
#         data[line.split(' ')[3].strip()] = 1
#     else:
#         data[line.split(' ')[3].strip()] += 1
# print("测试数据条数：",len(lines))
# print(data)
# for k,v in data.items():
#     all_sum += v
# print(all_sum)
#
# for k,v in data.items():
#     if "Negative" == k:
#         continue
#     x_data.append(k)
#     y_data.append(v)
# # 绘柱状图
# # plt.bar(x=x_data, height=y_data, label='Sentence_Length', color='steelblue', alpha=0.8)
# plt.bar(x=x_data, height=y_data, color='steelblue', alpha=0.8,width=0.5)
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # for x, y in enumerate(y_data):
# #     plt.text(x, y, '%s' % y, ha='center', va='bottom')
#
# # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# for x, y in enumerate(y_data):
#     plt.text(x, y, '%s' % y, ha='center', va='bottom')
# # 设置标题
# plt.title("ACE 2005 数据分布柱状图")
# # 为两条坐标轴设置名称
# plt.xlabel("关系类别")
# plt.ylabel("数量")
# # 显示图例
# plt.legend()
# plt.show()


def draw(x_label,y_label,file_path,title):
    all_sum = 0

    print(file_path + "分布：")
    data = dict()
    lines = open(file_path, 'r', encoding='utf-8').readlines()
    for k,line in enumerate(lines):
        # if len(line.split(' ')) != 4:
        #     print(k,line)
        # print(line.split(' ')[3].strip())

        # print(sentence)
        if line.split(' ')[3].strip() not in data:
            data[line.split(' ')[3].strip()] = 1
        else:
            data[line.split(' ')[3].strip()] += 1
    print(file_path, len(lines))
    print(data)
    data = sorted(data.items(),key=lambda item:item[0], reverse=False)
    print(data)
    for k in data:
        all_sum += k[1]
    print(all_sum)
    x_data = []
    y_data = []

    for k in data:
        if "Negative" == k[0]:
            continue
        x_data.append(k[0])
        y_data.append(k[1])
    # 绘柱状图
    # plt.bar(x=x_data, height=y_data, label='Sentence_Length', color='steelblue', alpha=0.8)
    plt.bar(x=x_data, height=y_data, color='steelblue', alpha=0.8, width=0.5)

    # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    for x, y in enumerate(y_data):
        plt.text(x, y, '%s' % y, ha='center', va='bottom')
    # 设置标题
    plt.title(title)
    # 为两条坐标轴设置名称
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 显示图例
    plt.legend()
    plt.show()

file_path = './data/__Result_nolap.txt'
x_label = '关系类别'
y_label = '数量'
title = 'ACE 2005 关系类别总分布图'
draw(x_label,y_label,file_path,title)

file_path = './data/train.txt'
x_label = '关系类别'
y_label = '数量'
title = 'ACE 2005 训练集关系类别分布图'
draw(x_label,y_label,file_path,title)

file_path = './data/dev.txt'
x_label = '关系类别'
y_label = '数量'
title = 'ACE 2005 验证集关系类别分布图'
draw(x_label,y_label,file_path,title)

file_path = './data/test.txt'
x_label = '关系类别'
y_label = '数量'
title = 'ACE 2005 测试集关系类别分布图'
draw(x_label,y_label,file_path,title)


