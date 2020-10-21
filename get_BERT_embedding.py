import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

ckpt_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict = reader.get_variable_to_shape_map()  # 读取 ckpt中的参数的维度的
# emd = param_dict['bert/embeddings/word_embeddings']
emd = reader.get_tensor('bert/embeddings/word_embeddings')  # 得到ckpt中指定的tensor
print(len(emd))
print(emd[:5])
param = np.array(emd)
np.save('chinese_L-12_H-768_A-12/bert_embed.txt', param)
'''
from tensorflow.python.tools import inspect_checkpoint as chkp
chkp.print_tensors_in_checkpoint_file(file_name="./bert_model.ckpt", 
                                      tensor_name = 'bert/embeddings/word_embeddings', 
                                      all_tensors = True, 
                                      all_tensor_names=True) #
'''