import torch
import h5py
from random import randint
import numpy as np


class KITTY(torch.utils.data.Dataset):
    def __init__(self, path, with_mvsec=False):
        super(KITTY).__init__()
        self.path = path
        self.file = h5py.File(path, "r")
        self.length = []
        for i in range(132):
            self.length.append(self.file[f"pred_{i}"].shape[0])
        self.len = sum(self.length)

    def __getitem__(self, idx):
        for i in range(132):
            if idx - self.length[i] >= 0:
                idx -= self.length[i]
            else:
                file_id = i
                break

        h = randint(0, 256)
        w = randint(0, 1136)

        pred_image = torch.Tensor(self.file[f"pred_{file_id}"][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))
        next_image = torch.Tensor(self.file[f"next_{file_id}"][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))

        event_image = torch.Tensor(np.stack([
            self.file[file_id][f"0_{file_id}"][idx][h:h + 256, w:w + 256],
            self.file[file_id][f"1_{file_id}"][idx][h:h + 256, w:w + 256],
            self.file[file_id][f"2_{file_id}"][idx][h:h + 256, w:w + 256],
            self.file[file_id][f"3_{file_id}"][idx][h:h + 256, w:w + 256]
        ]))

        return pred_image, next_image, event_image, 0

    def __len__(self):
        return self.len


class MVSEC(torch.utils.data.Dataset):
    def __init__(self, path):
        super(MVSEC).__init__()
        self.path = path
        self.file = h5py.File(path, "r")

    def __getitem__(self, idx):
        h = 2 #randint(0, 4)
        w = 45 #randint(0, 90)

        flow = self.file['flow'][idx][:, h:h + 256, w:w + 256]
        pred_image = torch.Tensor(self.file['pred'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))
        next_image = torch.Tensor(self.file['next'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))
        event_image = torch.Tensor(np.stack([
            self.file['0'][idx][h:h + 256, w:w + 256],
            self.file['1'][idx][h:h + 256, w:w + 256],
            self.file['2'][idx][h:h + 256, w:w + 256],
            self.file['3'][idx][h:h + 256, w:w + 256]
        ]))

        return pred_image, next_image, event_image, flow

    def __len__(self):
        return self.file['0'].shape[0]

class RAW(torch.utils.data.Dataset):
    def __init__(self, path):
        super(MVSEC).__init__()
        self.path = path
        self.file = h5py.File(path, "r")

    def __getitem__(self, idx):
        h = 2 #randint(0, 4)
        w = 45 #randint(0, 90)

        flow = self.file['flow'][idx][:, h:h + 256, w:w + 256]
        event_image = torch.Tensor(np.stack([
            self.file['0'][idx][h:h + 256, w:w + 256],
            self.file['1'][idx][h:h + 256, w:w + 256],
            self.file['2'][idx][h:h + 256, w:w + 256],
            self.file['3'][idx][h:h + 256, w:w + 256]
        ]))

        return event_image, flow

    def __len__(self):
        return self.file['flow'].shape[0]