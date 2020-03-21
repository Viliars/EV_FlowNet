from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from loss import photometric_loss, smoothness_loss
import h5py
from net import Model
from vis_utils import vis_all

path = Path("../data/")


class MVSEC(torch.utils.data.Dataset):
    def __init__(self, path):
        super(MVSEC).__init__()
        self.path = path

    def __getitem__(self, idx):
        events_array = {"15": [], "50": [], "80": [], "mvsec": []}
        event_images = {"15": [], "50": [], "80": [], "mvsec": []}
        pred_images = []
        next_images = []

        pred_images.append([np.array(Image.open(self.path / f"images/{idx}.png"))])
        next_images.append([np.array(Image.open(self.path / f"images/{idx + 1}.png"))])

        for key in events_array.keys():
            events = h5py.File(self.path / f"event_images/{key}/{idx}to{idx + 1}.hdf5", "r")
            x = np.array(events['x'])
            y = np.array(events['y'])
            t = np.array(events['t'])
            p = np.array(events['p'])
            events_array[key].append([x, y, t, p])

            event_image = np.zeros((4, 256, 256))
            for j in range(x.shape[0]):
                if p[j] == 1:
                    event_image[0, y[j], x[j]] += 1
                    event_image[2, y[j], x[j]] = t[j]
                if p[j] == -1:
                    event_image[1, y[j], x[j]] += 1
                    event_image[3, y[j], x[j]] = t[j]

            event_images[key].append(event_image)

        for key in event_images.keys():
            event_images[key] = torch.Tensor(event_images[key])

        return torch.Tensor(pred_images), torch.Tensor(next_images), events_array, event_images

    def __len__(self):
        return 1000


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model()
model = model.to(device)
model.load_state_dict(torch.load("data/model/model.pth"))
imsize = 256, 256
model.eval()

mvsec = MVSEC(path)

batch_size = 1
losses = {"15": [], "50": [], "80": [], "mvsec": []}
for i in tqdm(range(1000)):
    result = {"15": [], "50": [], "80": [], "mvsec": []}

    pred_images, next_images, events_array, event_images = mvsec[i]

    pred_images = pred_images.to(device)
    next_images = next_images.to(device)

    for key in event_images.keys():
        event_images[key] = event_images[key].to(device)

    for key in losses.keys():
        flow = model.forward(event_images[key])
        loss = photometric_loss(pred_images, next_images, flow) + 0.5 * smoothness_loss(flow)
        result[key] = flow[0]
        losses[key].append(loss.item())

    Image.fromarray(np.vstack([vis_all(result['15'], events_array['15'][0], pred_images[0][0]),
                               vis_all(result['50'], events_array['50'][0], pred_images[0][0]),
                               vis_all(result['80'], events_array['80'][0], pred_images[0][0])])
                    ).save(path / "result/image_{:010d}.jpg".format(i))

for key in losses.keys():
    print(key, np.average(losses[key]))



