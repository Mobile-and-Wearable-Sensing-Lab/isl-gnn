import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class SingleHandH5Dataset_MobileLastN(Dataset):
    def __init__(self, root_dir, label_file, n=60, debug=False, is_lstm=False, is_whisper=False, is_gnn=False, padding="default"):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        self.n = n
        self.is_whisper = is_whisper
        self.is_gnn = is_gnn
        self.padding = padding



        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                label = label_folder
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.h5'):
                        self.files.append(os.path.join(label_path, file_name))
                        self.labels.append(label)

        with open(label_file) as f:
            self.label_order = [i.strip() for i in f.readlines()]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_order)}
        self.labels = [self.label_to_idx[label] for label in self.labels]

        self.debug = debug
        self.is_lstm = is_lstm

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as h5_file:
            data = h5_file['intermediate'][:][:, :, :, :] #since we only have 1 hand remove that dimension
        
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # if not self.is_lstm:
        if self.padding == "default":
            # New logic
            # if data_tensor.shape[0] > self.n:
            #     # Center crop
            #     total = data_tensor.shape[0]
            #     start = (total - self.n) // 2
            #     data_tensor = data_tensor[start:start + self.n]

            # Old Logic
            if data_tensor.shape[0] > self.n:
                data_tensor = data_tensor[-self.n:]

            # New logic
            # if data_tensor.shape[0] < self.n:
            #     pad_num = self.n - data_tensor.shape[0]
            #     left_pad = pad_num // 2
            #     right_pad = pad_num - left_pad
            #     shape_suffix = list(data_tensor.shape[1:])
            #
            #     left = torch.zeros((left_pad, *shape_suffix), device=data_tensor.device, dtype=data_tensor.dtype)
            #     right = torch.zeros((right_pad, *shape_suffix), device=data_tensor.device, dtype=data_tensor.dtype)
            #
            #     data_tensor = torch.cat([left, data_tensor, right], dim=0)

            # Old Logic
            if data_tensor.shape[0] < self.n:
                middle_idx = data_tensor.shape[0] // 2
                pad_num = self.n - data_tensor.shape[0]

                data_tensor = torch.cat([
                    data_tensor[:middle_idx, :],
                    data_tensor[middle_idx:middle_idx+1, :].repeat([pad_num, *[1 for i in data_tensor.shape[1:]]]),
                    data_tensor[middle_idx:, :]
                ])

        elif self.padding == "linear-interp":
            data_tensor = data_tensor.permute(1, 2, 0)

            # Resample using linear interpolation
            resampled_tensor = F.interpolate(data_tensor, size=self.n, mode='linear', align_corners=False)

            # Remove the extra dimensions (reverse the unsqueeze)
            data_tensor = resampled_tensor.squeeze(0)
            data_tensor = data_tensor.permute(2, 0, 1)
    
        if self.is_lstm:
            data_tensor = data_tensor.view(data_tensor.size(0), -1)
        elif self.is_whisper:
            data_tensor = data_tensor.view(data_tensor.size(0), data_tensor.size(1), -1)
        elif self.is_gnn:
            pass
        else:
            data_tensor = data_tensor.view(1, data_tensor.size(0), -1)


        # One Hot Encoding
        label_tensor = torch.zeros(len(self.label_order), dtype=torch.float32)
        label_tensor[self.labels[idx]] = 1.0

        # None One Hot Encoding 
        # label_tensor = torch.zeros(1, dtype=torch.int32)
        # label_tensor[0] = self.labels[idx]
        if self.debug:
            return data_tensor, label_tensor, self.files[idx]
        return data_tensor, label_tensor
