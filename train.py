from pathlib import Path
import torch
import numpy as np
from net import Model
from datasets import KITTY, MVSEC, RAW
from tqdm import tqdm
from loss import photometric_loss, smoothness_loss
from flow_loss import flow_error_dense
from torch.utils.tensorboard import SummaryWriter
import random
import paths

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

data_path = Path(paths.data)
models_path = Path(paths.models)

train = KITTY(data_path/"kitty.hdf5", with_mvsec=False)
train_loader = torch.utils.data.DataLoader(train, batch_size=20, num_workers=1, shuffle=True, pin_memory=True)
raw1 = RAW(data_path/"raw1.hdf5")
raw1_loader = torch.utils.data.DataLoader(raw1, batch_size=20, num_workers=1, pin_memory=True)
raw2 = RAW(data_path/"raw2.hdf5")
raw2_loader = torch.utils.data.DataLoader(raw2, batch_size=20, num_workers=1, pin_memory=True)
raw3 = RAW(data_path/"raw3.hdf5")
raw3_loader = torch.utils.data.DataLoader(raw3, batch_size=20, num_workers=1, pin_memory=True)

writer = SummaryWriter()

device = torch.device('cuda:0')
model = Model()
model = model.to(device)
imsize = 256, 256

print(f"TrainSize = {len(train)}")
print(f"KITTYSize = {train.len}")
print(f"Raw1Size = {len(raw1)}")
print(f"Raw2Size = {len(raw2)}")
print(f"Raw3Size = {len(raw3)}")

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.8)

model.train()

for epoch in range(100):
    print(f"------ EPOCH {epoch} ------")
# -------------------------- TRAIN --------------------------
    train_losses = []
    for i_batch, sample_batched in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        pred_images, next_images, event_images, _ = sample_batched

        pred_images = pred_images.to(device)
        next_images = next_images.to(device)
        event_images = event_images.to(device)

        flow = model.forward(event_images)
        loss = photometric_loss(pred_images, next_images, flow) + 0.5 * smoothness_loss(flow)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    scheduler.step()
    print(f"Loss/train = {np.mean(train_losses)}")
    writer.add_scalar('Loss/train', np.mean(train_losses), epoch)

    model.eval()
# -------------------------- RAW1 ---------------------------

    raw1_AEE = []
    raw1_percent = []

    with torch.no_grad():
        for i_batch, sample_batched in tqdm(enumerate(raw1_loader)):
            event_images, gt_flow = sample_batched
            event_mask = torch.sum(event_images[:, :2, ...], dim=1)
            event_images = event_images.to(device)

            flow = model.forward(event_images)

            flow = flow.cpu()

            for i in range(flow.shape[0]):
                AEE, percent_AEE, _ = flow_error_dense(gt_flow[i], flow[i], event_mask[i])
                raw1_AEE.append(AEE)
                raw1_percent.append(percent_AEE)
    
    print(f"Raw1/AEE = {np.mean(raw1_AEE)}")
    print(f"Raw1/percent = {np.mean(raw1_percent)}")
    writer.add_scalar('Raw1/AEE', np.mean(raw1_AEE), epoch)
    writer.add_scalar('Raw1/percent', np.mean(raw1_percent), epoch)
    
    # -------------------------- RAW2 ---------------------------

    raw2_AEE = []
    raw2_percent = []

    with torch.no_grad():
        for i_batch, sample_batched in tqdm(enumerate(raw2_loader)):
            event_images, gt_flow = sample_batched
            event_mask = torch.sum(event_images[:, :2, ...], dim=1)
            event_images = event_images.to(device)

            flow = model.forward(event_images)

            flow = flow.cpu()

            for i in range(flow.shape[0]):
                AEE, percent_AEE, _ = flow_error_dense(gt_flow[i], flow[i], event_mask[i])
                raw2_AEE.append(AEE)
                raw2_percent.append(percent_AEE)
    
    print(f"Raw2/AEE = {np.mean(raw2_AEE)}")
    print(f"Raw2/percent = {np.mean(raw2_percent)}")
    writer.add_scalar('Raw2/AEE', np.mean(raw2_AEE), epoch)
    writer.add_scalar('Raw2/percent', np.mean(raw2_percent), epoch)
    
    # -------------------------- RAW3 ---------------------------
    
    raw3_AEE = []
    raw3_percent = []

    with torch.no_grad():
        for i_batch, sample_batched in tqdm(enumerate(raw3_loader)):
            event_images, gt_flow = sample_batched
            event_mask = torch.sum(event_images[:, :2, ...], dim=1)
            event_images = event_images.to(device)

            flow = model.forward(event_images)

            flow = flow.cpu()

            for i in range(flow.shape[0]):
                AEE, percent_AEE, _ = flow_error_dense(gt_flow[i], flow[i], event_mask[i])
                raw3_AEE.append(AEE)
                raw3_percent.append(percent_AEE)
    
    print(f"Raw3/AEE = {np.mean(raw3_AEE)}")
    print(f"Raw3/percent = {np.mean(raw3_percent)}")
    writer.add_scalar('Raw3/AEE', np.mean(raw3_AEE), epoch)
    writer.add_scalar('Raw3/percent', np.mean(raw3_percent), epoch)
    
    print(f"Test/AEE = {np.mean([np.mean(raw1_AEE), np.mean(raw2_AEE), np.mean(raw3_AEE)])}")
    print(f"Test/percent = {np.mean([np.mean(raw1_percent), np.mean(raw2_percent), np.mean(raw3_percent)])}")
    writer.add_scalar('Test/AEE', np.mean([np.mean(raw1_AEE), np.mean(raw2_AEE), np.mean(raw3_AEE)]), epoch)
    writer.add_scalar('Test/percent', np.mean([np.mean(raw1_percent), np.mean(raw2_percent), np.mean(raw3_percent)]), epoch)



    model.train()

writer.close()
torch.save(model.state_dict(), models_path/"15.pth")

