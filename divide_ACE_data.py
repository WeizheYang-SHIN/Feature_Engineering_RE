import random
all_data = './data/__Result_nolap.txt'
train_data = './data/train.txt'
dev_data = './data/dev.txt'
test_data = './data/test.txt'
train = open(train_data,'w',encoding='utf-8')
dev = open(dev_data,'w',encoding='utf-8')
test = open(test_data,'w',encoding='utf-8')

lines = open(all_data,'r',encoding='utf-8').readlines()
random.shuffle(lines)
length = len(lines)
for line in lines[0:int(length * 0.6)]:
    train.write(line)
for line in lines[int(length * 0.6):int(length*0.8)]:
    dev.write(line)
for line in lines[int(length * 0.8):]:
    test.write(line)