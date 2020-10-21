data_dir = './data/__Result_nolap.txt'
tags_dir = './data/tags.txt'
relations = set()
lines = open(data_dir,'r',encoding = 'utf-8')
for line in lines:
    relation = line.split(' ')[3]
    relations.add(relation)

tags_wr = open(tags_dir,'w',encoding = 'utf-8')
for relation in relations:
    tags_wr.write(relation)