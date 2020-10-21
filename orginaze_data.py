# -*- coding:utf-8 -*-
"""
@project:relaiton_extraction
@author:yangwz
@time:2019/10/22-17:18
"""

file_data = './data/__Result_nolap.txt'
lines = open(file_data,'r',encoding='utf-8').readlines()
for line in lines:
    print(line.split(' ')[3].strip())