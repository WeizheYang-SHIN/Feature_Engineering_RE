import os
import datetime
start_time = datetime.datetime.now()
print('Start excuting main_predict.py')
while True:
    end_time = datetime.datetime.now()
    if (end_time - start_time).seconds/60>=5:

        start_time = datetime.datetime.now()
        check_model_name = open('./output/models/checkpoint', 'r', encoding='utf-8').readline()
        print('Start predictint with model----------' + str(check_model_name))
        os.system('python main_predict.py')

