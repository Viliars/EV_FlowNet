import numpy as np
import torch

"""
Calculates per pixel flow error between flow_pred and flow_gt. 
event_img is used to mask out any pixels without events (are 0).
"""
def flow_error_dense(flow_gt, flow_pred, event_img):
    event_mask = event_img > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(~np.isinf(flow_gt[0, :, :]), ~np.isinf(flow_gt[1, :, :])),
        np.linalg.norm(flow_gt, axis=0) > 0)

    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask)).type(torch.bool)

    gt_masked = flow_gt[:, total_mask]
    pred_masked = flow_pred[:, total_mask]

    # Average endpoint error.
    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    n_points = EE.shape[0]
    AEE = np.mean(EE)

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    return AEE, percent_AEE, n_points