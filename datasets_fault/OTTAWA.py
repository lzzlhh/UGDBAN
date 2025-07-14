import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets_fault.SequenceDatasets import dataset
from datasets_fault.sequence_aug import *
from tqdm import tqdm

# Digital data was collected at 12,000 samples per second
signal_size = 1024
dataname= {0:["H-A-2", "I-A-2", "O-A-2", "B-A-2", "C-A-2"],
           1:["H-B-2", "I-B-2", "O-B-2", "B-B-2", "C-B-2"],
           2:["H-C-2", "I-C-2", "O-C-2", "B-C-2", "C-C-2"],
           3:["H-D-2", "I-D-2", "O-D-2", "B-D-2", "C-D-2"]}

datasetname = ["2 Data collected from a bearing with inner race fault", "3 Data collected from a bearing with outer race fault", "4 Data collected from a bearing with ball fault"
               ,"5 Data collected from a bearing with a combination of faults","1 Data collected from a healthy bearing"]
axis = ["Channel_1", "Channel_2"]

label = [i for i in range(0, 5)]

def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n==0:
               path1 =os.path.join(root,datasetname[4], dataname[N[k]][n])
            if n==1:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            if n == 2:
                path1 = os.path.join(root, datasetname[1], dataname[N[k]][n])
            if n == 3:
                path1 = os.path.join(root, datasetname[2], dataname[N[k]][n])
            if n == 4:
                path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n])
            data += data1
            lab +=lab1

    return [data, lab]

# signal_size = 1024
# dataname= {0:["H-A-3", "I-A-3", "O-A-3"],
#            1:["H-B-3", "I-B-3", "O-B-3"],
#            2:["H-C-3", "I-C-3", "O-C-3"],
#            3:["H-D-3", "I-D-3", "O-D-3"]}


# datasetname = ["2 Data collected from a bearing with inner race fault", "3 Data collected from a bearing with outer race fault","1 Data collected from a healthy bearing"]
# axis = ["Channel_1", "Channel_2"]
#
# label = [i for i in range(0, 3)]
#
# def get_files(root, N):
#     '''
#     This function is used to generate the final training set and test set.
#     root:The location of the data set
#     '''
#     data = []
#     lab =[]
#     for k in range(len(N)):
#         for n in tqdm(range(len(dataname[N[k]]))):
#             if n==0:
#                path1 =os.path.join(root,datasetname[2], dataname[N[k]][n])
#             if n==1:
#                 path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
#             if n == 2:
#                 path1 = os.path.join(root, datasetname[1], dataname[N[k]][n])
#             # if n == 3:
#             #     path1 = os.path.join(root, datasetname[2], dataname[N[k]][n])
#             # if n == 4:
#             #     path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
#             data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[n])
#             data += data1
#             lab +=lab1
#
#     return [data, lab]
#
def data_load(filename, axisname, label, max_samples=300):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    # datanumber = axisname.split(".")
    realaxis = axis[0]
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    sample_count = 0

    while end <= fl.shape[0] and sample_count < max_samples:
    # while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
        sample_count+=1

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class OTTAWA(object):
    num_classes = 10
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
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            # print(target_train[2])
            target_noshuffle= dataset(list_data=train_pd, transform=self.data_transforms['train'])
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


"""
    def data_split(self):

"""