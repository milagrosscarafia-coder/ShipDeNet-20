from utils.metrics import evaluate
from utils.decode import decode, xywh_to_xyxy
from utils.nms import postprocess
import os
import torch
import torch.nn as nn
from loss import ShipDeNetLoss
from utils.visualizacion import plotear
from matplotlib import pyplot as plt
import math


from tqdm import tqdm

from utils.visualizacion import plotear

from IPython.display import clear_output



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
    
    def fit(self, train_loader, val_loader, epochs=2000, eval_every=50, plot_every=50,
            out="best.pt", train_eval_loader=None):

        self.init_glorot()

        for head in [self.model.ssfp.layer18, self.model.ssfp.layer19, self.model.ssfp.layer20]:
            with torch.no_grad():
                head.point_conv.bias.view(3, self.vals)[:, 4] = math.log(0.01 / 0.99)   # score inicial ≈ 0.01
        
        best_loss, best_ap = float("inf"), -1.0
        self.tr_parts, self.val_parts, self.tr_metrics = [], [], []
        self.tr_losses, self.val_losses = [], []
        for epoch in tqdm(range(1, epochs + 1)):
            tr_loss, parts = self.train_one_epoch(train_loader)

            compute_ap = epoch % eval_every == 0
            val_loss, res = self.validate(val_loader, compute_ap)

            print(f"época {epoch} | train {tr_loss:.3f} {parts} | val {val_loss:.3f}")

            if res is not None:
                print(f"   val:   AP50 {res['AP50']:.4f}  P {res['precision']:.4f}  R {res['recall']:.4f}")
                self.val_parts.append(res)

                if train_eval_loader is not None:              # métricas en train, sin augmentation
                    _, res_tr = self.validate(train_eval_loader, compute_ap=True)
                    print(f"   train: AP50 {res_tr['AP50']:.4f}  P {res_tr['precision']:.4f}  R {res_tr['recall']:.4f}")
                    self.tr_metrics.append(res_tr)

                if res["AP50"] > best_ap:                      # mejor modelo por AP50
                    best_ap = res["AP50"]
                    torch.save(self.model.state_dict(), out)
                    print(f"   nuevo mejor AP50 {best_ap:.4f} (época {epoch})")

            if val_loss < best_loss:                           # mejor modelo por val loss (criterio del paper)
                best_loss = val_loss
                torch.save(self.model.state_dict(), out.replace(".pt", "_loss.pt"))

            self.tr_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            self.tr_parts.append(parts)

            if epoch % plot_every == 0:                        # DENTRO del loop
                # path = r"C:\Users\Usuario\Desktop\5to Semestre\Aprendizaje profundo con vision artificial\ShipDeNet\ShipDeNet-20\SSFP\losses_{epoch}.html"
                path = f"/content/drive/MyDrive/ShipDeNet_FF_FE/losses_{epoch}.html"
                fig = plotear(self.tr_losses, self.val_losses, self.tr_parts, self.val_parts,
                              path=path, desde=10, eval_every=eval_every,
                              tr_metrics=self.tr_metrics)
                clear_output(wait=True)
                fig.show()
        
    

        
        






