import h5py
from pathlib import Path
import paths

mvsec_path = Path(paths.mvsec)

file = h5py.File(mvsec_path/f"mvsec.hdf5", "r")

flow = file['flow'][::2]
preds = file['pred'][::2]
nexts = file['next'][::2]
event_images0 = file['0'][::2]
event_images1 = file['1'][::2]
event_images2 = file['2'][::2]
event_images3 = file['3'][::2]

f1 = h5py.File(mvsec_path/f"92.hdf5", "w")
f1.create_dataset("flow", data=flow)
f1.create_dataset("pred", data=preds)
f1.create_dataset("next", data=nexts)
f1.create_dataset("0", data=event_images0)
f1.create_dataset("1", data=event_images1)
f1.create_dataset("2", data=event_images2)
f1.create_dataset("3", data=event_images3)
f1.close()

flow = file['flow'][1::2]
preds = file['pred'][1::2]
nexts = file['next'][1::2]
event_images0 = file['0'][1::2]
event_images1 = file['1'][1::2]
event_images2 = file['2'][1::2]
event_images3 = file['3'][1::2]

f2 = h5py.File(mvsec_path/f"mvsec2.hdf5", "w")
f2.create_dataset("flow", data=flow)
f2.create_dataset("pred", data=preds)
f2.create_dataset("next", data=nexts)
f2.create_dataset("0", data=event_images0)
f2.create_dataset("1", data=event_images1)
f2.create_dataset("2", data=event_images2)
f2.create_dataset("3", data=event_images3)
f2.close()