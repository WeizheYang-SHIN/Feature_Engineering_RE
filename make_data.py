# -*- coding:utf-8 -*-
"""
@project:relaiton_extraction
@author:yangwz
@time:2019/12/19-21:16
"""
from random import shuffle
file_POST = open('./data/8_features_data/Post_8_features.txt','r',encoding='utf-8').readlines()
file_Text = open('./data/8_features_data/Text_8_features.txt','r',encoding='utf-8').readlines()
file_Turn = open('./data/8_features_data/Turn_8_features.txt','r',encoding='utf-8').readlines()
file_Positive = open('./data/8_features_data/Positive_8_features.txt','r',encoding='utf-8').readlines()

file_Train = open('./data/8_features_data/All_8_features_train.txt','w',encoding='utf-8')
file_Dev = open('./data/8_features_data/All_8_features_dev.txt','w',encoding='utf-8')
file_Test= open('./data/8_features_data/All_8_features_test.txt','w',encoding='utf-8')

all_lines = []

for line in file_POST:
    all_lines.append(line)
for line in file_Text:
    all_lines.append(line)
for line in file_Turn:
    all_lines.append(line)
for line in file_Positive:
    all_lines.append(line)
shuffle(all_lines)
length = len(all_lines)

train_lines = all_lines[0:int(length * 0.6)]
dev_lines = all_lines[int(length * 0.6):int(length * 0.8)]
test_lines = all_lines[int(length * 0.8):]

for line in train_lines:
    file_Train.write(line)
for line in dev_lines:
    file_Dev.write(line)
for line in test_lines:
    file_Test.write(line)