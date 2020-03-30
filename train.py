from pathlib import Path
import torch
import numpy as np
from net import Model
from datasets import KITTY, MVSEC
from tqdm import tqdm
from loss import photometric_loss, smoothness_loss
from flow_loss import flow_error_dense
from torch.utils.tensorboard import SummaryWriter

kitty_path = Path("")
mvsec_path = Path("")

train = KITTY(kitty_path)
train_loader = torch.utils.data.DataLoader(train, batch_size=16, num_workers=1, pin_memory=True)
test = MVSEC(mvsec_path)
test_loader = torch.utils.data.DataLoader(test, batch_size=16, num_workers=1, pin_memory=True)

writer = SummaryWriter()

device = torch.device('cuda:0')
model = Model()
model = model.to(device)
imsize = 256, 256


model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)

for epoch in range(300):
    print(f"------ EPOCH {epoch} ------")
    # TRAIN
    train_losses = []
    for i_batch, sample_batched in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()

        pred_images, next_images, event_images = sample_batched

        pred_images = pred_images.to(device)
        next_images = next_images.to(device)
        event_images = event_images.to(device)

        flow = model.forward(event_images)
        loss = photometric_loss(pred_images, next_images, flow) + 0.5 * smoothness_loss(flow)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    scheduler.step()
    print(f"EPOCH №{epoch} Loss/train = {np.mean(train_losses)}")
    writer.add_scalar('Loss/train', np.mean(train_losses), epoch)

    # TEST
    if epoch % 10 == 0 and epoch > 1:
        model.eval()

        test_losses_AEE = []
        test_losses_percent_AEE = []

        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(test_loader)):
                event_images, gt_flow = sample_batched
                event_mask = torch.sum(event_images[:, :2, ...], dim=1)
                event_images.to(device)

                flow = model.forward(event_images)

                flow.cpu()

                for i in range(flow.shape[0]):
                    AEE, percent_AEE, _ = flow_error_dense(gt_flow[i], flow[i], event_mask[i])
                    test_losses_AEE.append(AEE)
                    test_losses_percent_AEE.append(percent_AEE)

        print(f"EPOCH №{epoch} Loss/test_AEE = {np.mean(test_losses_AEE)}")
        print(f"EPOCH №{epoch} Loss/test_percent_AEE = {np.mean(test_losses_percent_AEE)}")

        writer.add_scalar('Loss/test_AEE', np.mean(test_losses_AEE), epoch)
        writer.add_scalar('Loss/test_percent_AEE', np.mean(test_losses_percent_AEE), epoch)

        model.train()
