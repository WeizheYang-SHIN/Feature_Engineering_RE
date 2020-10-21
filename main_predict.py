# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics
import collections
import csv
import datetime
# from bert4keras.backend import keras, set_gelu, K
import os
import bert_master.modeling as modeling
import bert_master.optimization as optimization
import bert_master.tokenization as tokenization
import tensorflow as tf
import pandas as pd
import jieba
import jieba.posseg
import platform
import numpy as np
import pickle
import json

flags = tf.flags

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

if os.name == 'nt':
    bert_path = './chinese_L-12_H-768_A-12/'
    root_path = './'
else:
    bert_path = './chinese_L-12_H-768_A-12/'
    root_path = './'

# %%
## Required parameters
flags.DEFINE_string(
    "data_dir", "./data/8_features_data_8_1_1",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", bert_path + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "relation_extraction", "The name of the task to train.")

flags.DEFINE_string("vocab_file", bert_path + "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./output/models",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", bert_path + "bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 185,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 64, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 64.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 512,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 512,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


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


def getNumClasses():
    tag_path = os.path.join(FLAGS.data_dir, "tags.txt")
    f = open(tag_path, 'r', encoding='utf-8')
    lines = f.readlines()
    label = []
    for line in lines:
        label.append(line.strip())
    f.close()
    return len(label)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, mark_index, feature_index, word_index, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.mark_index = mark_index
        self.feature_index = feature_index
        # self.head_index = head_index
        self.word_index = word_index


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 mark_index,
                 feature_index,
                 # head_index,
                 word_index,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.mark_index = mark_index,
        self.feature_index = feature_index,
        # self.head_index = head_index,
        self.word_index = word_index,
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        data = []
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
        Featute_7 = concate(Head_Entity1, Head_Entity2)
        Featute_8 = x_text12

        return_sentence = x_text13

        return (Featute_1, Featute_2, Featute_3, Featute_4, Featute_5, Featute_6, Featute_7, Featute_8, Entity_1,
                Entity_2, Type_Entity1, Type_Entity2,
                Subtype_Entity1, Subtype_Entity2, Head_Entity1, Head_Entity2), return_sentence, labels


class Relation_Extraction(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # self._read_tsv(os.path.join(data_dir, "train.json")), "train")
            self._read_tsv(os.path.join(data_dir, 'All_8_features_train.txt')), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'All_8_features_dev.txt')), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir)), "test")

    def get_labels(self):
        """See base class."""
        tag_path = os.path.join(FLAGS.data_dir, "tags.txt")
        f = open(tag_path, 'r', encoding='utf-8')
        lines = f.readlines()
        label = []
        for line in lines:
            label.append(line.strip())
        f.close()
        return label

    def _create_examples(self, lines, set_type):
        """得到的lines是一个元素为字典形式的列表，sentence是句子，labels是一个含有多标签的列表"""
        """Creates examples for the training and dev sets."""
        examples = []
        count_sum = 0
        for i in range(len(lines[1])):
            count_sum += 1
            # if count_sum == 50:
            #     break

            # Only the test set has a header
            # if set_type == "test" and i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            # if set_type == "test":
            #     text_a = tokenization.convert_to_unicode(line[1])
            #     label = "0"
            # else:
            #     text_a = tokenization.convert_to_unicode(line[3])
            #     label = tokenization.convert_to_unicode(line[1])
            n_grams = []
            n_grams_str = ''
            f1 = RightPos_Entity1_Type_Entity1 = lines[0][0][i]
            f2 = LeftPos_Entity1_Type_Entity1 = lines[0][1][i]
            f3 = RightPos_Entity2_Type_Entity2 = lines[0][2][i]
            f4 = LeftPos_Entity2_Type_Entity2 = lines[0][3][i]
            f5 = Type_Entity1_Type_Entity2 = lines[0][4][i]
            f6 = Subtype_Entity1_Subtype_Entity2 = lines[0][5][i]
            f7 = Head_Entity1_Head_Entity2 = lines[0][6][i]
            f8 = entityPosition = lines[0][7][i]
            entity1 = lines[0][8][i]
            entity2 = lines[0][9][i]
            entity1_type = lines[0][10][i]
            entity2_type = lines[0][11][i]
            entity1_subtype = lines[0][12][i]
            entity2_subtype = lines[0][13][i]
            entity1_head = lines[0][14][i]
            entity2_head = lines[0][15][i]
            sentence = lines[1][i]
            if ',' in entity1:
                entity1 = entity1.replace(', ', '')
            if ',' in entity2:
                entity2 = entity2.replace(', ', '')
            if ',' in sentence:
                sentence = sentence.replace(', ', '')
            for ii in range(len(sentence) - 2):
                n_grams.append(sentence[ii:ii + 3])
            for ii in n_grams:
                n_grams_str = n_grams_str + ' ' + ii
            n_grams_str = n_grams_str.strip()

            sentence_copy = sentence
            RightPos_Entity1 = f1.split('_')[0]
            LeftPos_Entity1 = f2.split('_')[0]
            RightPos_Entity2 = f3.split('_')[0]
            LeftPos_Entity2 = f4.split('_')[0]
            entity1_instead = ' <' + entity1_type + '_1> ' + entity1 + ' </' + entity1_type + '_1> '
            entity2_instead = ' <' + entity2_type + '_2> ' + entity2 + ' </' + entity2_type + '_2> '
            mark_1 = '<' + entity1_type + '_1>'
            mark_2 = '</' + entity1_type + '_1>'
            mark_3 = '<' + entity2_type + '_2>'
            mark_4 = '</' + entity2_type + '_2>'
            # 实验3:左 + <类型>实体1<类型> + 中  + <类型>实体2<类型>+ 右

            # if entity1 == '1' and entity2 == '1':
            #     entity1_index = 1 + sentence[1:].index(entity1)
            #     entity2_index = entity1_index + sentence[entity1_index + 1:].index(entity2)
            #     sentence1 = sentence[0]
            #     sentence2 = sentence[1:entity1_index+len(entity1)+1]
            #     sentence3 = sentence[entity1_index+len(entity1)+1:]
            #     sentence2 = sentence2.replace(entity1,entity1_instead)
            #     sentence3 = sentence3.replace(entity2,entity2_instead)
            #     sentence = sentence1 + sentence2 + sentence3
            if "1 死 1 重" in sentence:
                sentence = sentence.replace("1 死 1 重",
                                            '<' + entity1_type + '_1> ' + '1' + ' </' + entity1_type + '_1> ' + '死 ' +
                                            '<' + entity2_type + '_2> ' + '1' + ' </' + entity2_type + '_2> ' + '重')
            else:
                try:
                    if len(entity1) < len(entity2):
                        sentence = sentence.replace(entity2, entity2_instead)
                        sentence = sentence.replace(entity1, entity1_instead)
                    elif entity2 in entity1 and sentence.index(entity1) == sentence.index(entity2):
                        sentence = sentence.replace(entity1, entity1_instead)
                        sentence = sentence.replace(entity2, entity2_instead)
                    elif sentence.index(entity1) == sentence.index(entity2) and len(entity1) == len(entity2):
                        sentence = sentence.replace(entity1, entity1_instead)
                        sentence = sentence.replace(entity2, entity2_instead)
                    elif sentence.index(entity1) != sentence.index(entity2):
                        sentence = sentence.replace(entity1, entity1_instead)
                        sentence = sentence.replace(entity2, entity2_instead)
                    else:
                        sentence = sentence.replace(entity1, entity1_instead)
                        sentence = sentence[0:sentence.index(entity1_instead) + len(entity1_instead)] + sentence[
                                                                                                        sentence.index(
                                                                                                            entity1_instead) + len(
                                                                                                            entity1_instead):].replace(
                            entity2, entity2_instead)
                except:
                    print(sentence, '***1:', entity1, '***2:', entity2)
            all_features = f1 + " " + f2 + " " + f3 + " " + f4 + " " + f5 + " " + f6 + " " + f7 + " " + f8 + " " + sentence

            # all_features  = ''
            # atomic_features = []
            # atomic_features.append(entity1_head)
            # atomic_features.append(LeftPos_Entity1)
            # # atomic_features.append(entity1)
            # atomic_features.append(entity1_type)
            # atomic_features.append(entity1_subtype)
            # atomic_features.append(RightPos_Entity1)
            #
            # atomic_features.append(entity2_head)
            # atomic_features.append(LeftPos_Entity2)
            # # atomic_features.append(entity2)
            # atomic_features.append(entity2_type)
            # atomic_features.append(entity2_subtype)
            # atomic_features.append(RightPos_Entity2)
            # atomic_features.append(n_grams_str)
            # atomic_features.append(f8)
            # atomic_features.append(f1)
            # atomic_features.append(f2)
            # atomic_features.append(f3)
            # atomic_features.append(f4)
            # atomic_features.append(f5)
            # atomic_features.append(f6)
            # atomic_features.append(f7)
            # atomic_features.append(f8)
            # atomic_features.append(n_grams_str)
            # for auto_f in atomic_features:
            #     all_features = all_features + ' ' + auto_f
            all_features = all_features.strip()
            if count_sum <= 10:
                print('输入实例', count_sum, ":", all_features)
            # print('*************************',all_features)
            # all_features = f1 + ' ' + f2 + ' ' + f3 + ' ' + f4 + ' ' + f5 + ' ' + f6 + ' ' + f7 + ' ' + f8 + ' ' + entity1_instead + ' 0 ' + entity2_instead + ' 0 ' + sentence

            # entity1 = line.split(' ')[0]
            # entity2 = line.split(' ')[1]
            # sentence = line.split(' ')[2]
            # if len(entity1) < len(entity2):
            #     sentence = sentence.replace(entity1, '')
            #     sentence = sentence.replace(entity2, '')
            # else:
            #     sentence = sentence.replace(entity2, '')
            #     sentence = sentence.replace(entity1, '')
            text_a = tokenization.convert_to_unicode(all_features)
            mark_index = []
            feature_index = []
            head_index = []
            n_gram_index = []
            word_index = []

            # print(all_features)
            # print(mark_1)
            all_features_list = all_features.split(" ")
            mark_index.append(str(all_features_list.index(mark_1) + 1))
            mark_index.append(str(all_features_list.index(mark_2) + 1))
            mark_index.append(str(all_features_list.index(mark_3) + 1))
            mark_index.append(str(all_features_list.index(mark_4) + 1))
            f7_length = len(f7.split(' '))
            feature_index.append(str(1))
            feature_index.append(str(2))
            feature_index.append(str(3))
            feature_index.append(str(4))
            feature_index.append(str(5))
            feature_index.append(str(6))
            for mid_f in range(0,f7_length):
                feature_index.append(str(7 + mid_f))
            feature_index.append(str(f7_length + 7))
            pre_length_str = f1 + " " + f2 + " " + f3 + " " + f4 + " " + f5 + " " + f6 + " " + f7 + " " + f8
            pre_length_str_list = pre_length_str.split(" ")
            for k,word in enumerate(all_features_list[len(pre_length_str_list):-1]):
                word_index.append(str(k + len(pre_length_str_list) + 1))
            if len(word_index) < 100:
                for add_word in range(0,(100 - len(word_index))):
                    word_index.append(str(len(all_features.split(' '))))
            if len(word_index) > 100:
                word_index = word_index[0:100]


            # head_index.append(str(all_features.index(f7)))
            # for iii in range(1, len(head_index) + 1):
            #     head_index.append(str(all_features.index(f7) + iii))

            # n_gram_index.append(str(all_features.index(n_grams_str)))
            # for iii in range(1, len(n_grams) + 1):
            #     n_gram_index.append(str(all_features.index(n_grams_str) + iii))
            # text_a = tokenization.convert_to_unicode(entity1 + '0' + entity2 + '0' + sentence)
            # text_b = tokenization.convert_to_unicode(line.split(' ')[2])
            labels = []

            # print('**************', lines[2][i])
            labels.append(tokenization.convert_to_unicode(lines[2][i]))
            # labels.append(tokenization.convert_to_unicode(line.split(' ')[3].strip()))
            # labels = tokenization.convert_to_unicode(line.split(' ')[1])
            # for label in line['labels']:
            #     label = tokenization.convert_to_unicode(label)
            #     labels.append(label)
            mark_index = tokenization.convert_to_unicode(" ".join(mark_index))
            feature_index = tokenization.convert_to_unicode(" ".join(feature_index))
            # head_index = tokenization.convert_to_unicode(" ".join(head_index))
            # n_gram_index = tokenization.convert_to_unicode(" ".join(n_gram_index))
            word_index = tokenization.convert_to_unicode(" ".join(word_index))
            # mark_index = " ".join(mark_index)
            # feature_index = " ".join(feature_index)
            # head_index = " ".join(head_index)
            # n_gram_index = " ".join(n_gram_index)
            # if "##" in n_gram_index:
            #     print(n_gram_index)
            examples.append(
                InputExample(guid=guid, text_a=text_a, mark_index=mark_index, feature_index=feature_index, word_index=word_index, text_b=None, label=labels))
            # (self, guid, text_a, mark_index, feature_index, head_index, n_gram_index, text_b=None, label=None)
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        # mark_index, feature_index, head_index, n_gram_index
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    # mark_index = tokenizer.tokenize(example.mark_index)
    feature_index = tokenizer.tokenize(example.feature_index)
    mark_ids = []
    for mark_id in example.mark_index.split(' '):
        mark_ids.append(str(mark_id))
    mark_index = mark_ids

    # head_ids = []
    # for head_id in example.head_index.split(' '):
    #     head_ids.append(str(head_id))
    # head_index = head_ids
    # print("******:",example.n_gram_index)

    word_ids = []
    for word_id in example.word_index.split(' '):
        word_ids.append(str(word_id))
    word_index = word_ids
    # print("******:", example.n_gram_index)
    # n_gram_index = tokenizer.tokenize(example.n_gram_index)
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    mark_list = []
    mark_file = './data/8_features_data_8_1_1/marks.txt'
    ms = open(mark_file,'r',encoding='utf-8').readlines()
    for m in ms:
        mark_list.append(m.strip())
    tokens_c = tokenizer.tokenize(example.text_a)
    tokens_a = []
    remember_list = []
    flag = 0
    word = ''
    for each_f in tokens_c:
        if each_f == '<' or flag == 1:
            remember_list.append(each_f)
            if each_f == '>':
                flag = 0
                word += each_f
                if word not in mark_list:
                    for origin_mark in remember_list:
                        tokens_a.append(origin_mark)
                    word = ''
                    remember_list = []
                else:
                    tokens_a.append(word)
                    word = ''
            else:
                if "##" in each_f:
                    each_f =  each_f.replace('##','')
                flag = 1
                word += each_f
        else:
            tokens_a.append(each_f)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    mark_index = []
    feature_index = []
    word_index = []
    for id,seg in enumerate(tokens):
        feature_index.append(str(id))
        if seg in ['1','2','3','4']:
            break

    for id_2,seg in enumerate(tokens):
        if seg in mark_list:
            mark_index.append(id_2)

    for id_word,seg in enumerate(tokens[id + 1:]):
        if id_word + id + 1 in mark_index:
            continue
        else:
            word_index.append(id_word + id + 1)

    feature_length = len(feature_index)
    if feature_length > 41:
        feature_index = feature_index[0:41]
    if feature_length < 41:
        for add_id in range(41-feature_length):
            feature_index.append(0)

    mark_length = len(mark_index)
    if mark_length > 4:
        mark_index = mark_index[0:4]
    if mark_length < 4:
        for add_id in range(4-mark_length):
            mark_index.append(0)

    word_length = len(word_index)
    if word_length > 140:
        word_index = word_index[0:140]
    if word_length < 140:
        for add_id in range(140-word_length):
            word_index.append(0)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    ##label是一个列表
    # label_id = label_map[example.label]

    label_id = [0] * len(label_map)
    if len(example.label) > 0:
        for label in example.label:
            label_index = label_map[label]
            label_id[label_index] = 1

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (str(example.label), str(label_id)))
        tf.logging.info("mark_index: %s" % " ".join([str(x) for x in mark_index]))
        tf.logging.info("feature_index: %s" % " ".join([str(x) for x in feature_index]))
        # tf.logging.info("head_index: %s" % " ".join([str(x) for x in head_index]))
        tf.logging.info("word_index: %s" % " ".join([str(x) for x in word_index]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        mark_index=mark_index,
        feature_index=feature_index,
        # head_index=head_index,
        word_index=word_index,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        label_ids = feature.label_id
        features["label_ids"] = create_int_feature(label_ids)

        # features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        # numbers = [ int(x) for x in numbers ]
        feature.mark_index = list(feature.mark_index)
        feature.mark_index = [int(x) for x in feature.mark_index[0]]

        feature.feature_index = list(feature.feature_index)
        feature.feature_index = [int(x) for x in feature.feature_index[0]]

        # feature.head_index = list(feature.head_index)
        # feature.head_index = [int(x) for x in feature.head_index[0]]

        feature.word_index = list(feature.word_index)
        feature.word_index = [int(x) for x in feature.word_index[0]]
        # data_tensor= tf.convert_to_tensor(data_numpy)

        features["mark_index"] = create_int_feature(feature.mark_index)
        features["feature_index"] = create_int_feature(feature.feature_index)
        # features["head_index"] = create_int_feature(feature.head_index)
        features["word_index"] = create_int_feature(feature.word_index)
        # mark_index, feature_index, head_index, n_gram_index

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([getNumClasses()], tf.int64),  ##固定label_ids长度
        "mark_index": tf.FixedLenFeature([4], tf.int64),
        "feature_index": tf.FixedLenFeature([41], tf.int64),
        # "head_index": tf.FixedLenFeature([5], tf.int64),
        "word_index": tf.FixedLenFeature([140], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model_original(bert_config, is_training, input_ids, input_mask, segment_ids,
                          labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    # output_layer = model.get_pooled_output()  # 主干模型获得的模型输出
    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value  # bert的输出

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # 分类层
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


class CNN_ATTENTION():
    def __init__(self, embedding, labels, num_classes, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0001):
        self.embedding_size = embedding_size
        self.embedded_chars = embedding
        self.num_classes = num_classes
        self.dropout_keep_prob = 0.5
        self.input_y = labels

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=False):
                conv = tf.layers.conv1d(self.embedded_chars, num_filters, filter_size, name='conv1d')
                pooled = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # self.probability = tf.nn.softmax(self.scores, name="probability")
            self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def pro(self):

        return self.loss, self.losses, self.scores, self.probabilities

    def attention(self, x_i, x, index):
        """
        Attention model for Neural Machine Translation
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """

        e_i = []
        c_i = []
        for output in x:
            output = tf.reshape(output, [-1, self.embedding_size])
            atten_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_W), tf.matmul(output, self.attention_U)))
            e_i_j = tf.matmul(atten_hidden, self.attention_V)
            e_i.append(e_i_j)
        e_i = tf.concat(e_i, axis=1)
        # e_i = tf.exp(e_i)
        alpha_i = tf.nn.softmax(e_i)
        alpha_i = tf.split(alpha_i, self.sequence_length, 1)

        # i!=j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output)
                c_i.append(c_i_j)
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.sequence_length - 1, self.embedding_size])
        c_i = tf.reduce_sum(c_i, 1)
        return c_i


def create_cnn_attention_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                               labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    embedding = model.get_sequence_output()  # 主干模型获得的模型输出
    # used = tf.sign(tf.abs(input_ids))
    # seq_length = tf.reduce_sum(used, reduction_indices=1)
    # seq_length = embedding.shape  #[batch_size, embedding_size]

    embedding_size = embedding.shape[-1].value

    labels = tf.cast(labels, tf.float32)

    cnn = CNN_ATTENTION(embedding=embedding, labels=labels, num_classes=num_labels,
                        embedding_size=embedding_size, filter_sizes=[2, 3, 4, 5], num_filters=128)

    loss, per_example_loss, logits, probabilities = cnn.pro()

    return (loss, per_example_loss, logits, probabilities)


class CNN_model():
    def __init__(self, embedding, embedding_mark, embedding_feature,embedding_word, num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        self.embedding_size = embedding_size
        self.embedding = embedding
        self.embedding_mark = embedding_mark
        self.embedding_feature = embedding_feature
        self.embedding_word = embedding_word
        self.num_classes = num_classes
        self.dropout_keep_prob = 0.7
        # self.input_y = labels
        initializer = tf.keras.initializers.glorot_normal
        l2_loss = tf.constant(0.0)
        # self.embedding_chars = tf.expand_dims(self.embedding_chars, -1)
        # self.pooled_outputs1 = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.device('/gpu:2'):
        #         # Convolution Layer
        #         self.conv1 = tf.layers.conv2d(self.embedding_chars, num_filters, [filter_size, embedding_size],
        #                                       kernel_initializer=initializer(), activation=tf.nn.relu)
        #         self.conv = tf.squeeze(self.conv1, 2, name='conv')
        #         # Maxpooling over the outputs
        #         # pooled1 = tf.nn.max_pool(conv1, ksize=[1, sequence_length-filter_size+1, 1, 1],
        #         #                         strides=[1, 1, 1, 1], padding='VALID', name="pool1")
        #         self.max_pool = tf.reduce_max(self.conv, 1, name='max_pool')
        #         # self.max_pool = tf.expand_dims(self.max_pool, 1)
        #         # # Attention
        #         # with tf.variable_scope('attention-%s' % i):
        #         #     self.attn, self.alphas = attention(self.max_pool)
        #
        #         self.pooled_outputs1.append(self.max_pool)
        #
        # # Combine all the pooled features
        # # num_filters_total = num_filters * len(filter_sizes)
        # # self.h_pool1 = tf.concat(pooled_outputs1, 3)
        # # self.h_pool_flat1 = tf.reshape(self.h_pool1, [-1, num_filters_total])
        #
        # self.h_pool1 = tf.concat(self.pooled_outputs1, 1)
        #
        # # with tf.variable_scope('dense'):
        # #     self.dense = tf.layers.dense(self.final_outputs, 128)
        # # Add dropout
        # with tf.variable_scope("dropout"):
        #     self.h_drop = tf.nn.dropout(self.h_pool1, self.dropout_keep_prob)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv-max-pooling-%s' % filter_size):
                conv_1 = tf.layers.conv1d(self.embedding_word, num_filters, 3, name='conv1d_1')
                conv_2 = tf.layers.conv1d(self.embedding_mark, num_filters, filter_size, name='conv1d_2')
                conv_3 = tf.layers.conv1d(self.embedding_feature, num_filters, 1, name='conv1d_3')
                # conv_4 = tf.layers.conv1d(self.embedding_feature, num_filters, filter_size + 1, name='conv1d_4')
                conv = tf.concat([conv_1,conv_2,conv_3],1)
                pooled = tf.reduce_max(conv, reduction_indices=[1], name='max-pooling')
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # with tf.name_scope("output"):
        #   W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        #   b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #   l2_loss += tf.nn.l2_loss(W)
        #   l2_loss += tf.nn.l2_loss(b)
        #   self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        #   # self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #   self.probability = tf.nn.softmax(self.scores, name="probability")
        #   # self.probabilities = tf.nn.sigmoid(self.scores, name="probabilities")
        #
        # with tf.name_scope("loss"):
        #   self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        #   # self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.probability, labels=self.input_y)
        #   self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

    def pro(self):
        # return self.loss, self.losses, self.scores, self.probability
        return self.h_drop


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()  # 主干模型获得的模型输出

    hidden_size = output_layer.shape[-1].value  # bert的输出大小

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  # 分类层
        logits = tf.nn.bias_add(logits, output_bias)

        """在此处改为多标签分类的sigmoid激活函数"""
        # probabilities = tf.nn.sigmoid(logits)  # 该值是输出的预测结果
        probabilities = tf.nn.softmax(logits)  # 该值是输出的预测结果
        labels = tf.cast(labels, tf.float32)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # 可以使用focal loss损失函数解决样本不均衡的问题！！！！！！！！！！！！！！
        # def focal_loss(logits, labels, gamma = 2.0):
        #     '''
        #     :param logits:  [batch_size, n_class]
        #     :param labels: [batch_size]
        #     :return: -(1-y)^r * log(y)
        #     '''
        #     softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        #     labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        #     prob = tf.gather(softmax, labels)
        #     weight = tf.pow(tf.subtract(1., prob), gamma)
        #     loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        #     return loss

        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)

# def batch_gather(params, indices):
#     """params.shape=[b, n, d]，indices.shape=[b]
#     从params的第i个序列中选出第indices[i]个向量，返回shape=[b, d]。
#     """
#     indices = K.cast(indices, 'int32')
#     batch_idxs = K.arange(0, K.shape(indices)[0])
#     indices = K.stack([batch_idxs, indices], 1)
#     return tf.gather_nd(params, indices)

def batch_gather(params, indexs,length):
    tensors = []
    for i in range(length):
        indices = indexs[:, i]
        range_ = tf.range(0, tf.shape(indices)[0])
        indices = tf.stack([range_, indices], 1)
        tensors.append(indices)
    indices = tf.stack(tensors, 1)
    return tf.gather_nd(params, indices)


def create_model_cnn(bert_config, is_training, input_ids, input_mask, segment_ids,
                     labels, mark_index, feature_index, word_index, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    embedding = model.get_sequence_output()
    # embedding = model.get_pooled_output()
    print("111111111", embedding.shape)
    embedding_mark = batch_gather(embedding, mark_index, 4)
    # embedding_mark = tf.gather_nd(embedding, [[[0, 0], [0, 3], [0,5], [0,6]], [[1,2], [1,4], [1,2], [1,3]]])
    print("111111111", embedding_mark.shape)
    embedding_feature = batch_gather(embedding, feature_index, 41)
    print("111111111", embedding_feature.shape)
    embedding_word = batch_gather(embedding, word_index,140)
    print("111111111", embedding_word.shape)
    # embedding_head = batch_gather(embedding, head_index,5)
    # print("111111111", embedding_head.shape)
    # embedding_n_gram = batch_gather(embedding, n_gram_index,120)
    # print("111111111", embedding_n_gram.shape)
    # embedding_mark.append(embedding[0])
    # print('embedding_mark:', embedding_mark)
    # for id in mark_index:
    #     embedding_mark.append(embedding[0])
    # for id in feature_index:
    #     embedding_feature.append(embedding[1])
    # for id in head_index:
    #     embedding_head.append(embedding[id])
    # for id in n_gram_index:
    #     embedding_n_gram.append(embedding[id])

    hidden_size = embedding.shape[-1].value

    cnn = CNN_model(embedding=embedding,embedding_mark=embedding_mark,embedding_feature=embedding_feature,
                    embedding_word=embedding_word,num_classes=num_labels,
                    embedding_size=hidden_size, filter_sizes=[1,3], num_filters=50)

    # loss, per_example_loss, logits, probabilities = cnn.pro()
    output_layer = cnn.pro()
    hidden_size1 = output_layer.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size1],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.7)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        labels = tf.to_float(labels)

        per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, train_examples):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        mark_index = features["mark_index"]
        # print(mark_index)
        feature_index = features["feature_index"]
        # head_index = features["head_index"]
        word_index = features["word_index"]

        # for ids in train_examples.mark_index[0:32]:
        #     mark_index.append(train_examples.mark_index[ids]['mark_index'])
        # for ids in train_examples.feature_index[0:32]:
        #     mark_index.append(train_examples.feature_index[ids]['feature_index'])
        # for ids in train_examples.head_index[0:32]:
        #     mark_index.append(train_examples.head_index[ids]['head_index'])
        # for ids in train_examples.n_gram_index[0:32]:
        #     mark_index.append(train_examples.n_gram_index[ids]['n_gram_index'])
        # mark_index  = train_examples.mark_index
        # feature_index = train_examples.feature_index
        # head_index = train_examples.head_index
        # n_gram_index = train_examples.n_gram_index
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        ####create_model是原始的bert加全连接，create_cnn_attention_model在bert后增加cnn和attention
        (total_loss, per_example_loss, logits, probabilities) = create_model_cnn(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, mark_index, feature_index, word_index,
            # head_index,
            # n_gram_index,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,                                                                                init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.nn.sigmoid(logits)

                def multi_label_hot(predictions, threshold=0.5):
                    predictions = tf.cast(predictions, tf.float32)
                    threshold = float(threshold)
                    return tf.cast(tf.greater(predictions, threshold), tf.int64)

                one_hot_prediction = multi_label_hot(predictions)
                accuracy = tf.metrics.accuracy(tf.cast(one_hot_prediction, tf.int32), label_ids)
                # predictions = tf.equal(tf.cast(tf.greater_equal(tf.nn.sigmoid(logits),0.5),tf.int32),tf.cast(label_ids,tf.int32))
                # accuracy = tf.reduce_mean(tf.cast(predictions,tf.float32))
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def result_labels(data_type):
    tf.logging.set_verbosity(tf.logging.INFO)
    # mark_index_1 =,
    # feature_index_1 =,
    # head_index_1 =,
    # n_gram_index_1 =

    processors = {
        "relation_extraction": Relation_Extraction,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    print(label_list)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        train_examples=train_examples)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        # predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_examples = processor.get_test_examples(data_type)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        pred_labels = []
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            # output_file = './data/output.json'
            # output_file = open(output_file,'w')
            i = 0
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                # print(prediction)
                # print(type(prediction))
                prediction = prediction['probabilities']
                # print(prediction)
                # print(type(prediction))
                predicted_labels = []
                predicted_scores = []
                count = 0
                for i, score in enumerate(prediction):
                    if score >= 0.5:
                        predicted_scores.append(score)
                        predicted_labels.append(i)
                        count += 1
                pred_labels.append(predicted_labels)
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)
    return pred_labels


def generate_pred_file(tags_list, tags, inf_path, outf_path):
    # save_PRF = open('./output/save_PRF_' + str(PRF_name) + '.txt', 'a+', encoding='utf-8')
    with open(inf_path, 'r', encoding='utf-8') as inf, open(outf_path, 'w', encoding='utf-8') as outf:
        pred_labels = result_labels(inf_path)
        true_labels = []
        sum = 0
        count = 0
        count_sum = 0
        count_sum = 0
        for line in inf.readlines():
            count_sum += 1
            # if count_sum == 50:
            #     break
            line = line.strip()
            line = line.split(' ')
            # print(line)
            sent = line[-2].strip()
            test_label = line[-1].strip()
            # print('tags:',tags)
            # print('test_label:',test_label)
            true_labels.append(tags[test_label])
            pred_label = pred_labels[sum]
            sum += 1
            label_names = []
            for label in pred_label:
                label_names.append(tags_list[label])
            relation = label_names
            outf.write(sent + ' ' + str(relation) + '\n')

        preds = []
        for labels in pred_labels:
            if len(labels) != 0:
                preds.append(labels[0])
            else:
                preds.append(0)
        # preds = pred_labels
        truths = true_labels
        cls_results = metrics.classification_report(truths, preds, labels=[1, 2, 3, 4, 5, 6],
                                                    digits=4,
                                                    target_names=['PHYS', 'ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE',
                                                                  'PER-SOC'])
        # print(cls_results)
        wr_line = open('./output/PRF.txt', 'a+', encoding='utf-8')
        check_p_name = open('./output/models/checkpoint', 'r', encoding='utf-8').readline()
        wr_line.write(check_p_name + '\n')
        wr_line.write(cls_results)
        wr_line.close()
        wr_line = open('./output/this_PRF.txt', 'w', encoding='utf-8')
        wr_line.write(cls_results)
        wr_line.close()
        lines = open('./output/this_PRF.txt', 'r', encoding='utf-8').readlines()
        f1 = 0
        for line in lines:
            print(line)
            if 'macro avg' in line:
                f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
                    line.split('\t')[0].split('    ')[2].strip()) / (
                             float(line.split('\t')[0].split('    ')[1].strip()) + float(
                         line.split('\t')[0].split('    ')[2].strip()))
        print('The True F1 of ' + check_p_name + ' is ', f1)


def count_max_f(file):
    lines = open(file, 'r', encoding='utf-8').readlines()
    max_f = 0
    line_count = 0
    max_line_count = 0
    k = -12

    for line in lines:
        line_count += 1
        if 'macro avg' in line:
            f1 = 2 * float(line.split('\t')[0].split('    ')[1].strip()) * float(
                line.split('\t')[0].split('    ')[2].strip()) / (
                         float(line.split('\t')[0].split('    ')[1].strip()) + float(
                     line.split('\t')[0].split('    ')[2].strip()))
            if (f1 > max_f):
                max_f = f1
                max_line_count = line_count

    for i in range(13):
        print(lines[max_line_count + k])
        k += 1

    print('The true max f1:', max_f)
    print('最大值所在行占总行数的：', max_line_count / len(lines))


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    tags_list = []
    tags = dict()
    i = 0
    flag = 0
    inf_path = './data/8_features_data_8_1_1/All_8_features_test.txt'
    outf_path = './output/__Predict_test.txt'
    with open('./data/8_features_data_8_1_1/tags.txt', 'r', encoding='utf-8') as tagf:
        for line in tagf.readlines():
            tags_list.append(line.strip())
            tags[line.strip()] = i
            i += 1
    generate_pred_file(tags_list, tags, inf_path, outf_path)
    count_max_f('./output/PRF.txt')

# count_max_f('./output/save_PRF_' + str(start_time) + '.txt')
