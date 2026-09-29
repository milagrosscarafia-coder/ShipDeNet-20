import torch
from torch.utils.data import DataLoader, Subset, random_split

from train import ShipDeNetTrain
from shipBaseline import ShipDeNet
from read_data_set.anchors import best_of_runs, load_train_wh
from read_data_set.rdata import SSDD_BBox_coco

from models.layers import DSConv
from utils.visualizacion import visualizar, plotear
from tqdm import tqdm

import os
BASE = os.path.join("Official-SSDD-OPEN", "Official-SSDD-OPEN", "BBox_SSDD", "coco_style")
img_dir = os.path.join(BASE, "images", "train")
ann_file = os.path.join(BASE, "annotations", "train.json")


EPOCHS, BATCH, LR, VALS = 2000, 32, 1e-3, 5


def collate(batch):
    images, boxes = zip(*batch)
    return torch.stack(images), list(boxes)


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


OVERFIT = False     # True: prueba con 8 imágenes. False: entrenamiento real.

if OVERFIT:
    EPOCHS, BATCH = 300, 8

# ---- datos
dataset = SSDD_BBox_coco(img_dir, ann_file, size=160)

if OVERFIT:
    train_set = val_set = Subset(dataset, range(8))   # las mismas 8 imágenes
else:
    n_val = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(0))


train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, collate_fn=collate)

images, _ = next(iter(train_loader))
print(images.shape, images.dtype, images.min().item(), images.max().item(), images.mean().item())
# ---- anchors (una vez; después se pueden cargar de anchors.pt)
wh = load_train_wh(img_dir, ann_file)
anchors, mean_iou = best_of_runs(wh, k=9)
print(f"IoU medio de los anchors: {mean_iou:.4f}")
torch.save(anchors, "anchors.pt")

# ---- modelo, optimizador, scheduler
model = ShipDeNet(c_in=1, use_ff=True, use_fe=True, use_ssfp=True,
                    out_ch=3 * VALS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
total_iters = EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda it: (1 - it / total_iters) ** 0.9)

# ---- entrenamiento
trainer = ShipDeNetTrain(model, anchors, optimizer, scheduler, device, vals=VALS)
trainer.fit(train_loader, val_loader, epochs=EPOCHS,  eval_every=25 if OVERFIT else 50, plot_every=50, out="best.pt")


# cargar el MEJOR modelo guardado, no los pesos de la última época
model.load_state_dict(torch.load("best.pt", map_location=device))

visualizar(model, val_loader, anchors, vals=VALS, device=device,
           nombre="ShipDeNet-20 (validación)")

