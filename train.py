from pathlib import Path
import torch
import numpy as np
from net import Model
from datasets import KITTY
from tqdm import tqdm
from loss import photometric_loss, smoothness_loss
from torch.utils.tensorboard import SummaryWriter

path = Path("/hpcfs/GRAPHICS2/23m_pri/data/")

dataset = KITTY(path)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=1, pin_memory=True)

writer = SummaryWriter()

device = torch.device('cuda:0')
model = Model()
model = model.to(device)
imsize = 256, 256


model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)

for epoch in range(30):
    print(f"------ EPOCH {epoch} ------")
    losses = []
    for i_batch, sample_batched in tqdm(enumerate(dataset_loader)):
        optimizer.zero_grad()

        pred_images, next_images, event_images = sample_batched

        pred_images.to(device)
        next_images.to(device)
        event_images.to(device)

        flow = model.forward(event_images)
        loss = photometric_loss(pred_images, next_images, flow) + 0.5 * smoothness_loss(flow)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    writer.add_scalar('Loss/train', np.average(losses), epoch)

