import os
import pandas as pd
import polars as pl
import numpy as np
from torch.utils.data import Dataset
from utils.mypath import MyPath
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SREKPI(Dataset):
    base_folder = ""

    def __init__(
        self,
        fname,
        root=MyPath.db_root_dir("sre"),
        train=True,
        transform=None,
        sanomaly=None,
        mean_data=None,
        std_data=None,
        test_split: float = 0.3,
    ):
        super(SREKPI, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ["Normal", "Anomaly"]
        self.data = []
        self.targets = []
        wsz, stride = 240, 1

        self.mean, self.std = mean_data, std_data

        columns = [
          "dur_avg",
          "rps",
          "4xx",
          "5xx",
          "percent_above",
          "anomaly",
        ]

        file_path = os.path.join(self.root, self.base_folder, fname+'.parquet')
        temp = pl.read_parquet(file_path).select(columns).cast(pl.Float32).to_numpy()
        if self.train:
            known_anons = temp[:int(len(temp) * (1 - test_split)), -1]
            temp = temp[:int(len(temp) * (1 - test_split)), :1]
            
            if sanomaly.portion_len > 0:
                temp = temp[known_anons == 0]
                
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp , axis=0)
            self.targets = []
        else:
            temp = temp[int(len(temp) * (1 - test_split)):]
            self.targets =  temp[:, -1]

            temp = temp[int(len(temp) * (1 - test_split)):, :1]

            self.std[self.std == 0.0] = 1.0
            temp = (temp - self.mean) / self.std

        if len(temp.shape) == 1:
            temp = temp.reshape(-1, 1)

        self.data = np.asarray(temp)
        self.data, self.targets = self.convert_to_windows(wsz, stride)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st+w_size]
            if sum(self.targets[st:st+w_size]) > 0:
                lbl = 1
            else: lbl=0
                    
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        # ts_org = self.data[index]
        ts_org = torch.from_numpy(self.data[index]).float().to(device)  # cuda

        if len(self.targets) > 0:
            # target = self.targets[index].astype(int)
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long).to(device)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''
        
        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {'ts_org': ts_org, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
