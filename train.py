from pathlib import Path
import torch
from net import Model
from datasets import KITTY

path = Path("~/data/")

dataset = KITTY(path)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=1, pin_memory=True)

device = torch.device('cuda:0')
model = Model()
model = model.to(device)
imsize = 256, 256

print(len(dataset))

# model.train()
#
# optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
#
# losses = []
# images = []
# for epoch in range(30):
#     for i_batch, sample_batched in tqdm(enumerate(dataset_loader)):
#         optimizer.zero_grad()
#
#         pred_images, next_images, event_images = sample_batched
#
#         pred_images.to(device)
#         next_images.to(device)
#         event_images.to(device)
#
#         flow = model.forward(event_images)
#         loss = photometric_loss(pred_images, next_images, flow) + 0.5 * smoothness_loss(flow)
#
#         loss.backward()
#         optimizer.step()

