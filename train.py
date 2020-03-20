from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from loss import photometric_loss, smoothness_loss
import h5py
from pathlib import Path
from net import Model

path = Path("../data/")

event_images = []
pred_images = []
next_images = []

for i in tqdm(range(32)):
    pred_images.append([np.array(Image.open(path/f"images/{i}.png"))])
    next_images.append([np.array(Image.open(path/f"images/{i+1}.png"))])
    events = h5py.File(path/f"event_images/{i}to{i+1}.hdf5", "r")
    x = np.array(events['x'])
    y = np.array(events['y'])
    t = np.array(events['t'])
    p = np.array(events['p'])
    event_image = np.zeros((4, 256, 256))
    for i in range(x.shape[0]):
        if p[i] == 1:
            event_image[0, y[i], x[i]] += 1
            event_image[2, y[i], x[i]] = t[i]
        if p[i] == -1:
            event_image[1, y[i], x[i]] += 1
            event_image[3, y[i], x[i]] = t[i]
    event_images.append(event_image)

event_images = torch.Tensor(event_images)
pred_images = torch.Tensor(pred_images)
next_images = torch.Tensor(next_images)

device = torch.device('cuda:0')
model = Model(device)
imsize = 256, 256

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)

batch_size = 8

for epoch in range(10000):
    order = np.random.permutation(32)

    for start_index in range(0, 32, batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        event_batch = event_images[batch_indexes].to(device)
        pred_batch = pred_images[batch_indexes].to(device)
        next_batch = next_images[batch_indexes].to(device)

        flow = model.forward(event_batch)

        loss_value = photometric_loss(pred_batch, next_batch, flow) + 0.5 * smoothness_loss(flow)
        loss_value.backward()

        optimizer.step()

    print(f"epoch {epoch}, loss = {loss_value.item()}")
