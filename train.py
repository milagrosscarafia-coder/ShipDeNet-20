from utils.metrics import evaluate
from utils.decode import decode, xywh_to_xyxy
from utils.nms import postprocess
import torch
import torch.nn as nn
from loss import ShipDeNetLoss
from utils.visualizacion import plotear
from matplotlib import pyplot as plt

from tqdm import tqdm

from utils.visualizacion import plotear


IMG = 160 # dimension de la imagen 
STRIDES = [32, 16, 8] # mismo orden que [p32 (L/32), p16 (L/16), p8 (L/8)]
EPOCHS = 2000
BATCH_SIZE = 32 


class ShipDeNetTrain (nn.Module):
    def __init__(self, model, anchors, optimizer, scheduler, device, vals=5):
        super().__init__()

        self.model = model
        self.anchors = anchors.to(device)
        self.criterion = ShipDeNetLoss(anchors, vals=vals).to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.vals = vals

        

    def init_glorot(self): 
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def train_one_epoch(self, loader):
        self.model.train()                       # modo entrenamiento en cada época
        total = 0.0
        sums = {"xy": 0.0, "wh": 0.0, "obj": 0.0, "noobj": 0.0}
        for images, targets in loader:
            preds = self.model(images.to(self.device))
            loss, parts = self.criterion(preds, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total += loss.item()
            for k in sums:
                sums[k] += parts[k]
        n = len(loader)                          # fuera del loop de batches
        return total / n, {k: v / n for k, v in sums.items()}

    @torch.no_grad()
    def validate(self, loader, compute_ap=False):
        self.model.eval()
        total = 0.0
        all_dets, all_gts = [], []

        for images, targets in loader:

            preds = self.model(images.to(self.device))

            loss, _ = self.criterion(preds, targets)
            
            total += loss.item()

            if compute_ap:                       # adentro del loop: todos los batches
            
                dets = postprocess(decode(preds, self.anchors, self.vals), score_thr=0.001)
            
                all_dets += [d.cpu() for d in dets]
            
                all_gts += [xywh_to_xyxy(t.float()) for t in targets]
        
        result = evaluate(all_dets, all_gts) if compute_ap else None
        
        return total / len(loader), result

    def fit(self, train_loader, val_loader, epochs=2000, eval_every=50, plot_every=50, out="best.pt"):

        self.init_glorot()
        best = float("inf")

        self.tr_losses, self.val_losses = [], []
        self.tr_parts, self.val_parts = [], []

        for epoch in tqdm(range(1, epochs+1)):
            tr_loss, parts = self.train_one_epoch(train_loader)

            compute_ap = epoch % eval_every == 0
            
            val_loss, res = self.validate(val_loader, compute_ap)

            print(f"época {epoch} | train {tr_loss:.3f} {parts} | val {val_loss:.3f}")
            if res is not None:
                print(f"   AP50 {res['AP50']:.4f}  P {res['precision']:.4f}  R {res['recall']:.4f}")
                self.val_parts.append(res)
            if val_loss < best:
                best = val_loss
                torch.save(self.model.state_dict(), out)

            self.tr_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            self.tr_parts.append(parts)

            if epoch % plot_every == 0:
                path = f"/content/drive/MyDrive/ShipDeNet_0/losses_{epoch}.png"
                fig = plotear(self.tr_losses, self.val_losses, self.tr_parts, self.val_parts, path=path, desde=10)
                try:                             # si corre en un notebook, la muestra en vivo
                    from IPython.display import display
                    display(fig)
                except ImportError:
                    pass
                plt.close("all")
        
    

        
        






