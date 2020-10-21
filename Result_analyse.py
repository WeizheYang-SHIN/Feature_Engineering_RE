# -*- coding:utf-8 -*-
"""
@project:relaiton_extraction
@author:yangwz
@time:2019/12/13-16:04
"""
import os

file_path = './output/CNN'
lists = os.listdir(file_path)

def count_max_f(file):
    lines = open(file,'r',encoding='utf-8').readlines()
    max_f = 0
    line_count = 0
    max_line_count = 0
    k = -13

    for line in lines:
        # if 'macro avg' in line:
        #     print(line)
        #     print(line.split('\t')[0].split('    ')[1].strip())
        #     print(line.split('\t')[0].split('    ')[2].strip())
        #     print(line.split('\t')[0].split('    ')[3].strip())
        line_count += 1
        if 'macro avg' in line:
            f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(line.split('\t')[0].split('    ')[2].strip()) / (float(line.split('\t')[0].split('    ')[1].strip()) + float(line.split('\t')[0].split('    ')[2].strip()))
            if(f1 > max_f):
                max_f = f1
                max_line_count = line_count

    for i in range(14):
        print(lines[max_line_count + k])
        k += 1

    print('The true max f1:',max_f)

for file in lists:
    print(file)
    count_max_f(os.path.join(file_path,file))