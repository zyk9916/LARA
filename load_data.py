# 采用官方划分的数据集，进行预处理

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, train_csv, user_emb_matrix):
        self.train_csv = pd.read_csv(train_csv, header=None)                # 添加列索引
        # 分别赋值
        self.user = self.train_csv.loc[:, 0]
        self.item = self.train_csv.loc[:, 1]
        self.attr = self.train_csv.loc[:, 2]
        self.user_emb_matrix = pd.read_csv(user_emb_matrix, header=None)    # 添加列索引
        self.user_emb_values = np.array(self.user_emb_matrix[:])

    def __len__(self):
        return len(self.train_csv)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        user_emb = self.user_emb_values[user]
        # 处理属性，将字符串类型转换为整数
        attr = self.attr[idx][1:-1].split()
        attr = torch.tensor(list([int(item) for item in attr]), dtype=torch.long)
        attr = np.array(attr)
        return user, item, attr, user_emb