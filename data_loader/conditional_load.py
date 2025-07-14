import os
import importlib
import pandas as pd
from scipy.io import loadmat
import aug
import data_utils
import load_methods
from aug import sig_process,add_noise

# def get_files(root, dataset, faults, signal_size, condition=3,type='time'):
#     data, labels = [], []
#     data_load = getattr(load_methods, dataset)
#     func = getattr(sig_process, type)
#     for index, name in enumerate(faults):
#         data_dir = os.path.join(root, '%dHP' % condition, name)
#
#         for item in os.listdir(data_dir):
#             item_path = os.path.join(data_dir, item)
#             signal = data_load(item_path)
#
#             start, end = 0, signal_size
#             while end <= signal.shape[0]:
#                 x = func(signal[start:end])
#                 # x = x.reshape([1] + list(x.shape))  # get channel dimension若是时域，注释掉
#                 data.append(x)
#                 # data.append(signal[start:end])
#                 labels.append(index)
#                 start += signal_size
#                 end += signal_size
#     return data, labels
def get_files(root, dataset, faults, signal_size, condition=3, type='time', overlap=1):
    data, labels = [], []
    data_load = getattr(load_methods, dataset)
    func = getattr(sig_process, type)
    sample_count = 0  # 初始化样本计数器
    num_samples=250
    fault_sample_counts = {}
    for fault in faults:
        fault_sample_counts[fault] = 0
    for index, name in enumerate(faults):
        data_dir = os.path.join(root, '%dHP' % condition, name)
        for item in os.listdir(data_dir):
            if fault_sample_counts[name] >= num_samples:
                break
            item_path = os.path.join(data_dir, item)
            signal = data_load(item_path)
            # if fault_sample_count >= num_samples:
            #     break  # 达到每个fault所需样本数量后，提前结束循环

            step = int(signal_size * overlap)  # 计算重叠部分的
            start = 0  # 样本起始位置
            end = signal_size  # 样本终止位置

            while end <= signal.shape[0] and fault_sample_counts[name]  < num_samples:
            #while end <= signal.shape[0]:
                x = func(signal[start:end])
                data.append(x)
                labels.append(index)
                start += step  # 更新样本起始位置
                end = start + signal_size  # 更新样本终止位置
                # sample_count += 1  # 更新样本计数
                fault_sample_counts[name] += 1
    return data, labels
def data_transforms(normlize_type="-1-1"):
    transforms = {
        'train': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype(),
        ]),
        'val': aug.Compose([
            aug.Reshape(),
            aug.Normalize(normlize_type),
            aug.Retype(),
        ])
    }
    return transforms
# class dataset(object):
#
#     def __init__(self, data_dir, dataset, faults, signal_size, normlizetype, condition=2,
#                  balance_data=True, test_size=0.2):
#         self.balance_data = balance_data
#         self.test_size = test_size
#         self.num_classes = len(faults)
#         # self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition,type='time')
#         self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size,
#                                            condition=condition, type='time')
#         self.transform = data_transforms(normlizetype)
#
#     def data_preprare(self, source_label=None, is_src=False, random_state=1):
#
#         data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
#         data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
#         if is_src:
#             datasets_S = data_utils.dataset(list_data=data_pd, source_label=source_label, transform=self.transform['train'])
#             return datasets_S
#         else:
#             # train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size, num_classes=self.num_classes, random_state=random_state)
#             # train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label, transform=self.transform['train'])
#             datasets_T = data_utils.dataset(list_data=data_pd, source_label=source_label, transform=self.transform['val'] )
#             return  datasets_T
#
#     def data_preprare_val(self, source_label=None, is_src=False, random_state=1):
#         data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
#         data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
#         if is_src:
#             train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label,
#                                                transform=self.transform)
#             return train_dataset
#         else:
#             # train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size,
#             #                                                 num_classes=self.num_classes, random_state=random_state)
#             # train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label,
#             #                                    transform=self.transform['train'])
#             val_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label,
#                                              transform=self.transform['val'])
#             return val_dataset
class dataset(object):

    def __init__(self, data_dir, dataset, faults, signal_size, normlizetype, condition=2,
                 balance_data=True, test_size=0.2):
        self.balance_data = balance_data
        self.test_size = test_size
        self.num_classes = len(faults)
        # self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size, condition=condition,type='time')
        self.data, self.labels = get_files(root=data_dir, dataset=dataset, faults=faults, signal_size=signal_size,
                                           condition=condition, type='time')
        self.transform = data_transforms(normlizetype)

    def data_preprare(self, source_label=None, is_src=False, random_state=1):

        data_pd = pd.DataFrame({"data": self.data, "labels": self.labels})
        data_pd = data_utils.balance_data(data_pd) if self.balance_data else data_pd
        if is_src:
            train_dataset = data_utils.dataset(list_data=data_pd, source_label=source_label,
                                               transform=self.transform['train'])
            return train_dataset
        else:
            train_pd, val_pd = data_utils.train_test_split_(data_pd, test_size=self.test_size,
                                                            num_classes=self.num_classes, random_state=random_state)
            train_dataset = data_utils.dataset(list_data=train_pd, source_label=source_label,
                                               transform=self.transform['train'])
            val_dataset = data_utils.dataset(list_data=val_pd, source_label=source_label,
                                             transform=self.transform['val'])
            val_shuffle = data_utils.dataset(list_data=train_pd, source_label=source_label,
                                               transform=self.transform['val'])
            return train_dataset, val_dataset, val_shuffle