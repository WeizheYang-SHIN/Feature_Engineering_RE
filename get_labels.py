file_train = './data/8_features_data_8_1_1/All_8_features_train.txt'
file_dev = './data/8_features_data_8_1_1/All_8_features_dev.txt'
file_test = './data/8_features_data_8_1_1/All_8_features_test.txt'
mark_file = './data/8_features_data_8_1_1/marks.txt'
marked = set()

lines = open(file_train,'r',encoding='utf-8').readlines()
entity_types = ['VEH', 'WEA', 'GPE', 'PER', 'LOC', 'ORG', 'FAC']
for line in lines:
    entity_1_type = ''.join(line.split(' ')[6].strip())
    entity_2_type = ''.join(line.split(' ')[7].strip())
    if entity_2_type not in entity_types:
        entity_2_type = ''.join(line.split(' ')[8].strip())
    marked.add('<' + entity_1_type + '_1>')
    marked.add('</' + entity_1_type + '_1>')
    marked.add('<' + entity_2_type + '_2>')
    marked.add('</' + entity_2_type + '_2>')
print(len(marked),marked)

lines = open(file_dev,'r',encoding='utf-8').readlines()
entity_types = ['VEH', 'WEA', 'GPE', 'PER', 'LOC', 'ORG', 'FAC']
for line in lines:
    entity_1_type = ''.join(line.split(' ')[6].strip())
    entity_2_type = ''.join(line.split(' ')[7].strip())
    if entity_2_type not in entity_types:
        entity_2_type = ''.join(line.split(' ')[8].strip())
    marked.add('<' + entity_1_type + '_1>')
    marked.add('</' + entity_1_type + '_1>')
    marked.add('<' + entity_2_type + '_2>')
    marked.add('</' + entity_2_type + '_2>')
print(len(marked),marked)

lines = open(file_test,'r',encoding='utf-8').readlines()
entity_types = ['VEH', 'WEA', 'GPE', 'PER', 'LOC', 'ORG', 'FAC']
for line in lines:
    entity_1_type = ''.join(line.split(' ')[6].strip())
    entity_2_type = ''.join(line.split(' ')[7].strip())
    if entity_2_type not in entity_types:
        entity_2_type = ''.join(line.split(' ')[8].strip())
    marked.add('<' + entity_1_type + '_1>')
    marked.add('</' + entity_1_type + '_1>')
    marked.add('<' + entity_2_type + '_2>')
    marked.add('</' + entity_2_type + '_2>')
print(len(marked),marked)


wr_mark = open(mark_file,'w',encoding='utf-8')
for mark in marked:
    wr_mark.write(mark + '\n')
