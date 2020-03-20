from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from loss import photometric_loss, smoothness_loss
import h5py
from net import Model
from vis_utils import vis_all

path = Path("")

events_array = {"15": [], "50": [], "80": [], "mvsec": []}
event_images = {"15": [], "50": [], "80": [], "mvsec": []}
pred_images = []
next_images = []

for i in tqdm(range(1000)):
    pred_images.append([np.array(Image.open(path/f"images/{i}.png"))])
    next_images.append([np.array(Image.open(path/f"images/{i+1}.png"))])
    
    for key in events_array.keys():
        events = h5py.File(path/f"event_images/{key}/{i}to{i+1}.hdf5")
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

pred_images = torch.Tensor(pred_images)
next_images = torch.Tensor(next_images)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Model()
model.to(device)
model.load_state_dict(torch.load("data/model/model.pth"))
imsize = 256, 256


model.eval()

batch_size = 1
losses = {"15": [], "50": [], "80": [], "mvsec": []}
for i in tqdm(range(0, 1000, batch_size)):
    result = {"15": [], "50": [], "80": [], "mvsec": []}
    for key in losses.keys():
        flow = model.forward(event_images[key][i:i+batch_size])
        loss = photometric_loss(pred_images[i:i+batch_size],
                                next_images[i:i+batch_size], flow) + 0.5 * smoothness_loss(flow)
        result[key] = flow[0]
        losses[key].append(loss.item())
    
    Image.fromarray(np.vstack([vis_all(result['15'], events_array['15'][i], pred_images[i][0]),
               vis_all(result['50'], events_array['50'][i], pred_images[i][0]),
               vis_all(result['80'], events_array['80'][i], pred_images[i][0])])
                   ).save(path/"result/image_{:010d}.jpg".format(i))

for key in losses.keys():
    print(key, np.average(losses[key]))



