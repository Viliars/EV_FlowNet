import torch
import h5py
from random import randint
import numpy as np


class KITTY(torch.utils.data.Dataset):
    def __init__(self, path):
        super(KITTY).__init__()
        self.path = path
        self.length = []
        for i in range(92):
            file = h5py.File(path / "{:010d}.hdf5".format(i), "r")
            self.length.append(file['pred'].shape[0])
            file.close()
        self.len = sum(self.length)

    def __getitem__(self, idx):
        for i in range(92):
            if idx - self.length[i] >= 0:
                idx -= self.length[i]
            else:
                file_id = i
                break
        file = h5py.File(self.path / "{:010d}.hdf5".format(file_id), "r")

        h = randint(0, 256)
        w = randint(0, 1136)

        pred_image = torch.Tensor(file['pred'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))
        next_image = torch.Tensor(file['next'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))

        event_image = torch.Tensor(np.stack([
            file['0'][idx][h:h + 256, w:w + 256],
            file['1'][idx][h:h + 256, w:w + 256],
            file['2'][idx][h:h + 256, w:w + 256],
            file['3'][idx][h:h + 256, w:w + 256]
        ]))

        file.close()

        return pred_image, next_image, event_image

    def __len__(self):
        return self.len


class MVSEC(torch.utils.data.Dataset):
    def __init__(self, path):
        super(MVSEC).__init__()
        self.path = path
        self.file = h5py.File(path, "r")

    def __getitem__(self, idx):
        h = randint(0, 4)
        w = randint(0, 90)

        flow = self.file['flow'][idx][:, h:h + 256, w:w + 256]

        event_image = torch.Tensor(np.stack([
            self.file['0'][idx][h:h + 256, w:w + 256],
            self.file['1'][idx][h:h + 256, w:w + 256],
            self.file['2'][idx][h:h + 256, w:w + 256],
            self.file['3'][idx][h:h + 256, w:w + 256]
        ]))

        return event_image, flow

    def __len__(self):
        return 2203