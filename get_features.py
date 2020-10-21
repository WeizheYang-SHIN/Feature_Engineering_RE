import pandas as pd
import jieba
import jieba.posseg
file_train = './data/8_features_data_8_1_1/All_8_features_train.txt'
file_dev = './data/8_features_data_8_1_1/All_8_features_dev.txt'
file_test = './data/8_features_data_8_1_1/All_8_features_test.txt'
features_file = open('./data/8_features_data_8_1_1/features.txt','w',encoding='utf-8')

def getPOSleft(text):
    POS = []
    for words in text:
        words = words.strip()
        if words == '#' or len(words) == 0:
            POS.append('None')
        else:
            word = words[-1].strip()
            for word, flag in jieba.posseg.cut(word):
                POS.append(flag)
    return POS


def getPOSright(text):
    POS = []
    for words in text:
        words = words.strip()
        if words == '#' or len(words) == 0:
            POS.append('None')
        else:
            word = words[0].strip()
            for word, flag in jieba.posseg.cut(word):
                POS.append(flag)
    return POS


def concate(data_1, data_2):
    result = []
    for i in range(len(data_1)):
        result.append(data_1[i] + '_' + data_2[i])
    return result
def _read_tsv(input_file):
        data = []
        feature = set()
        entity_types = ['VEH', 'WEA', 'GPE', 'PER', 'LOC', 'ORG', 'FAC']
        lines = open(input_file, 'r', encoding='utf-8').readlines()
        for line in lines:
            entity_position = line.split(' ')[0].strip()  # 实体相对位置
            left = ' '.join(line.split(' ')[1].strip())  # 左边部分
            entity_1 = ' '.join(line.split(' ')[2].strip())  # 实体1
            middle = ' '.join(line.split(' ')[3].strip())
            entity_2 = ' '.join(line.split(' ')[4].strip())
            right = ' '.join(line.split(' ')[5].strip())
            entity_1_type = ''.join(line.split(' ')[6].strip())
            entity_2_type = ''.join(line.split(' ')[7].strip())
            entity_1_subtype = ''.join(line.split(' ')[8].strip())
            entity_2_subtype = ''.join(line.split(' ')[9].strip())
            if entity_2_type not in entity_types:
                entity_2_type = ''.join(line.split(' ')[8].strip())
                entity_1_subtype = ''.join(line.split(' ')[7].strip())
                entity_2_subtype = ''.join(line.split(' ')[9].strip())
            entity_1_head = ' '.join(line.split(' ')[10].strip())
            entity_2_head = ' '.join(line.split(' ')[11].strip())
            sentence = ' '.join(line.split(' ')[12].strip())
            relation = line.split(' ')[13].strip()
            if relation not in ['Negative', 'PHYS', 'ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC']:
                print(line)
            data.append([left, entity_1, middle, entity_2, right, entity_1_type, entity_2_type, entity_1_subtype,
                         entity_2_subtype, entity_1_head, entity_2_head, entity_position, sentence, relation])
            # data.append([id, sentence, relation])
        df = pd.DataFrame(data, columns=["left", "entity_1", "middle", "entity_2", "right", "entity1_Type",
                                         "entity2_Type", "entity1_Subtype",
                                         "entity2_Subtype", "entity1_Head", "entity2_Head", "entityPosition",
                                         "sentence", "relation"])
        x_text1 = df['left'].tolist()
        x_text2 = df['entity_1'].tolist()
        x_text3 = df['middle'].tolist()
        x_text4 = df['entity_2'].tolist()
        x_text5 = df['right'].tolist()
        x_text6 = df['entity1_Type'].tolist()
        x_text7 = df['entity2_Type'].tolist()
        x_text8 = df['entity1_Subtype'].tolist()
        x_text9 = df['entity2_Subtype'].tolist()
        x_text10 = df['entity1_Head'].tolist()
        x_text11 = df['entity2_Head'].tolist()
        x_text12 = df['entityPosition'].tolist()
        x_text13 = df['sentence'].tolist()
        labels = df['relation']

        Entity_1 = x_text2
        Entity_2 = x_text4
        Type_Entity1 = x_text6
        Type_Entity2 = x_text7
        Subtype_Entity1 = x_text8
        Subtype_Entity2 = x_text9
        Head_Entity1 = x_text10
        Head_Entity2 = x_text11
        LeftPos_Entity1 = getPOSleft(x_text1)
        RightPos_Entity1 = getPOSright(x_text3)
        LeftPos_Entity2 = getPOSleft(x_text3)
        RightPos_Entity2 = getPOSright(x_text5)

        Featute_1 = concate(RightPos_Entity1, Type_Entity1)
        Featute_2 = concate(LeftPos_Entity1, Type_Entity1)
        Featute_3 = concate(RightPos_Entity2, Type_Entity2)
        Featute_4 = concate(LeftPos_Entity2, Type_Entity2)
        Featute_5 = concate(Type_Entity1, Type_Entity2)
        Featute_6 = concate(Subtype_Entity1, Subtype_Entity2)
        # Featute_7 = concate(Head_Entity1, Head_Entity2)
        # Featute_8 = x_text12

        return_sentence = x_text13

        for f in Featute_1:
            feature.add(f.strip())
        for f in Featute_2:
            feature.add(f.strip())
        for f in Featute_3:
            feature.add(f.strip())
        for f in Featute_4:
            feature.add(f.strip())
        # for f in Featute_5:
        #     feature.add(f.strip())
        # for f in Featute_6:
        #     feature.add(f.strip())
        for f in feature:
            features_file.write(f.strip() + '\n')
        print(len(feature),feature)
_read_tsv(file_train)
_read_tsv(file_dev)
_read_tsv(file_test)
