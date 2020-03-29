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
            file = h5py.File(path / "{:010d}".format(i), "r")
            self.length.append(file['pred'].shape[0])
            file.close()
        self.len = sum(self.length)

    def __getitem__(self, idx):
        h = randint(0, 256)
        w = randint(0, 1136)
        pred_image = torch.Tensor(self.file['pred'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))
        next_image = torch.Tensor(self.file['next'][idx][h:h + 256, w:w + 256].reshape(1, 256, 256))

        event_image = torch.Tensor(np.stack([
            self.file['0'][idx][h:h + 256, w:w + 256],
            self.file['1'][idx][h:h + 256, w:w + 256],
            self.file['2'][idx][h:h + 256, w:w + 256],
            self.file['3'][idx][h:h + 256, w:w + 256]
        ]))

        return pred_image, next_image, event_image

    def __len__(self):
        return self.len