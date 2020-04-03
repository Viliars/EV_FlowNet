from pathlib import Path
import torch
import numpy as np
from net import Model
from datasets import KITTY, MVSEC
from tqdm import tqdm
from loss import photometric_loss, smoothness_loss
from flow_loss import flow_error_dense
from torch.utils.tensorboard import SummaryWriter
import paths

kitty_path = Path(paths.kitty)
mvsec_path = Path(paths.mvsec)
models_path = Path(paths.models)

test = MVSEC(mvsec_path)
test_loader = torch.utils.data.DataLoader(test, batch_size=16, num_workers=1, shuffle=True, pin_memory=True)

device = torch.device('cuda:0')
model = Model()
model = model.to(device)
imsize = 256, 256

print(f"TestSize = {len(test)}")

for epoch in range(122):
    # TEST
    if epoch % 10 == 0 and epoch > 1:
        print(f"------ EPOCH {epoch} ------")
        model.load_state_dict(torch.load(models_path/f"model{epoch}.pth", map_location=device))

        model.eval()

        test_losses_AEE = []
        test_losses_percent_AEE = []

        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(test_loader)):
                event_images, gt_flow = sample_batched
                event_mask = torch.sum(event_images[:, :2, ...], dim=1)
                event_images = event_images.to(device)

                flow = model.forward(event_images)

                flow = flow.cpu()

                for i in range(flow.shape[0]):
                    AEE, percent_AEE, _ = flow_error_dense(gt_flow[i], flow[i], event_mask[i])
                    test_losses_AEE.append(AEE)
                    test_losses_percent_AEE.append(percent_AEE)

        print(f"Loss/test_AEE = {np.mean(test_losses_AEE)}")
        print(f"Loss/test_percent_AEE = {np.mean(test_losses_percent_AEE)}")
