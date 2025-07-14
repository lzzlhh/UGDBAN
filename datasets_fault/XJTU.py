import os
import json
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets_fault.SequenceDatasets import dataset
from datasets_fault.sequence_aug import *
from tqdm import tqdm
from itertools import islice



#Digital data was collected at 12,000 samples per second
signal_size = 1024
# work_condition=['_10Nm-1000rpm.csv','_10Nm-2000rpm.csv','_10Nm-3000rpm.csv']
# work_condition=['1500-0.txt','1500-2.txt','1500-4.txt','1500-6.txt','1500-8.txt','1500-10.txt']
dataname= {0:[os.path.join('35Hz12kN','Bearing1_1','22.csv'),
              os.path.join('35Hz12kN','Bearing1_4','22.csv'),
              os.path.join('35Hz12kN','Bearing1_5','22.csv'),
              ],
         1:[os.path.join('37.5Hz11kN','Bearing2_2','22.csv'),
              os.path.join('37.5Hz11kN','Bearing2_3','22.csv'),
              os.path.join('37.5Hz11kN','Bearing2_1','22.csv'),
              ],

           }

label = [i for i in range(0, 3)]

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            path1 = os.path.join(root, dataname[N[k]][n])
            if n==0:
                data1, lab1 = data_load(path1,  label=label[n])
            else:
                data1, lab1 = data_load(path1, label=label[n])
            data += data1
            lab +=lab1

    return [data, lab]


# def data_load(filename, label):
#     '''
#     This function is mainly used to generate test data and training data.
#     filename:Data location
#     axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
#     '''
#     # datanumber = axisname.split(".")
#     max_samples = 300
#     realaxis = axis[7]
#     df = pd.read_csv(filename, delimiter='\t')
#     fl = df[realaxis]
#     data = []
#     lab = []
#     start, end = 0, signal_size
#     sample_count = 0
def data_load(filename, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    #--------------------
    f = open(filename, "r", encoding='gb18030', errors='ignore')
    fl = []
    # if  "ball_20_0.csv" in filename:
    #     for line in islice(f, 1, None):  # Skip the first 1 lines
    #         line = line.rstrip()
    #         word = line.split(",", 8)  # Separated by commas
    #         fl.append(eval(word[1]))  # Take a vibration signal in the x direction as input
    # else:
    df = pd.read_csv(filename) # 第13行开始为加速度数据

    fl = df["Horizontal_vibration_signals"]
    fl = fl.values.reshape(-1,1)
    # for line in islice(f, 1, None):  # Skip the first 1 lines
    #     line = line.rstrip()
    #     word = line.split(",", 8)  # Separated by \t
    #     fl.append(eval(word[5]))  # Take a vibration signal in the x direction as input
    #--------------------
    # fl = np.array(fl)
    # fl = fl.reshape(-1, 1)
    # print(fl.shape())
    data = []
    lab = []
    start, end = 0, signal_size
    sample_count = 0
    while end <= fl.shape[0] and sample_count < 1000:
    # while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
        sample_count+=1

    return data, lab
#--------------------------------------------------------------------------------------------------------------------
class XJTU(object):
    num_classes = 3
    inputchannel = 1
    def __init__(self, data_dir, transfer_task, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            class_number = [cla for cla in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cla))]
            # 排序，保证各平台顺序一致
            class_number.sort()
            # 生成类别名称以及对应的数字索引
            class_indices = dict((k, v) for v, k in enumerate(class_number))
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('class_indices.json', 'w') as json_file:
                json_file.write(json_str.encode('utf-8').decode('unicode_escape'))
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_noshuffle = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val, target_noshuffle
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val