import numpy as np
from skimage.color import hsv2rgb

def vis_flow(_flow):
    flow = _flow.permute(1, 2, 0).detach().numpy()
    mag = np.linalg.norm(flow, axis=2)
    a_mag = np.min(mag)
    b_mag = np.max(mag)

    ang = np.arctan2(flow[...,0], flow[...,1])
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros(list(flow.shape[:2]) + [3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag, 0, 255)
    hsv[:, :, 2] = ((mag - a_mag).astype(np.float32) * (255. / (b_mag - a_mag + 1e-32))).astype(np.uint8)
    flow_rgb = hsv2rgb(hsv)
    return 255 - (flow_rgb * 255).astype(np.uint8)

def vis_events(event_image):
    event_image_xy = event_image[0] + event_image[1]
    event_image_xy = np.array(event_image_xy, dtype=np.uint8)
    event_image_xy[event_image_xy > 0] = 255
    return np.tile(event_image_xy[..., None], (1, 1, 3))

    # res = np.zeros(imsize, dtype=np.uint8).ravel()
    # x, y = map(lambda x: x.astype(int), events[:2])
    # i = np.ravel_multi_index([y, x], imsize)
    # np.maximum.at(res, i, np.full_like(x, 255, dtype=np.uint8))
    # return np.tile(res.reshape(imsize)[..., None], (1, 1, 3))

def vis_image(image):
    return np.stack([image, image, image], axis=2).astype(np.uint8)

def collage(flow_rgb, events_rgb, image_rgb):
    return np.hstack([flow_rgb, events_rgb, image_rgb])

def vis_all(flow, event_image, image):
    return collage(vis_flow(flow), vis_events(event_image), vis_image(image))
