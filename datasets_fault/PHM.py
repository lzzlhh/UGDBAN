import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets_fault.SequenceDatasets import dataset
from datasets_fault.sequence_aug import *
from tqdm import tqdm


signal_size = 1024


# Case One
Case1 = ['helical 1',"helical 2","helical 3","helical 4","helical 5","helical 6"]

label1 = [i for i in range(6)]
#Case Two
Case2 = ['spur 1',"spur 2","spur 3","spur 4","spur 5","spur 6","spur 7","spur 8"]

label2 = [i for i in range(8)]

#working condition

WC = {0:"30hz"+"_"+"High"+"_1",
      1:"35hz"+"_"+"High"+"_1",
      2:"40hz"+"_"+"High"+"_1",
      3:"45hz"+"_"+"High"+"_1"}

#generate Training Dataset and Testing Dataset
def get_files(root, N):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []
    for k in range(len(N)):
        state1 = WC[N[k]]  # WC[0] can be changed to different working states
        for i in tqdm(range(len(Case2))):
            root1 = os.path.join("/tmp",root,Case2[i],Case2[i]+"_"+state1)
            datalist1 = os.listdir(root1)
            path1=os.path.join('/tmp',root1,datalist1[0])
            data1, lab1 = data_load(path1,label=label2[i])
            data += data1
            lab  += lab1
    return [data,lab]

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename,usecols=0)
    fl = fl.reshape(-1,1)
    data=[] 
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        # if start==0:
            # print(filename)
            # print(fl[start:end])
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class PHM(object):
    num_classes = 8    #Case 1 have 6 labels; Case 2 have 9 lables
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
            list_data = get_files(self.data_dir, self.target_N)

            ################################生成类别的json文件
            # class_number = [cla for cla in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cla))]
            # # 排序，保证各平台顺序一致
            # class_number.sort()
            # # 生成类别名称以及对应的数字索引
            # class_indices = dict((k, v) for v, k in enumerate(class_number))
            # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            # with open('class_indices.json', 'w') as json_file:
            #     json_file.write(json_str.encode('utf-8').decode('unicode_escape'))
            # ####################################################################
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


