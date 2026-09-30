import json, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt 
import matplotlib.patches as patches


class SSDD_BBox_coco():
    def __init__(self, img_dir, ann_file, size=160):
        coco = json.load(open(ann_file)) #cargamos las anotaciones
        self.img_dir, self.size = img_dir, size
        self.images = coco["images"]
        self.boxes = {im["id"]: [] for im in self.images} # se crea la lista vacia para cada imagen
        for a in coco["annotations"]: 
            self.boxes[a["image_id"]].append(a["bbox"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = Image.open(f"{self.img_dir}/{info['file_name']}").convert("L") # la convierte a blanco y negro
        sx, sy = self.size / info["width"], self.size/ info["height"] # calcula cuanto hay que redimensionar la imagen 
        img = TF.to_tensor(TF.resize(img, [self.size, self.size]))   # (1, 160, 160)
        b = torch.tensor(self.boxes[info["id"]], dtype=torch.float32).reshape(-1, 4)
        b[:, [0,2]] *= sx
        b[:, [1,3]] *= sy
        return img, b 

def print_imagen(img, b=None): 
    fig, ax = plt.subplots(1)

    ax.imshow(img.squeeze(), cmap='gray')
    if b is not None:
        for box in b:
            x , y, w, h = box
            rect = patches.Rectangle((x,y),w,h,linewidth=2,
                                    edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()

class AugSAR(torch.utils.data.Dataset):
    """Flips y rotaciones de 90° al azar, solo para entrenamiento.
    Cajas: [x, y, w, h] en píxeles de una imagen cuadrada de size x size."""
    def __init__(self, base, size=160):
        self.base, self.size = base, size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, b = self.base[i]
        b = b.clone().float()
        S = self.size
        if torch.rand(1) < 0.5:                        # flip horizontal
            img = img.flip(-1)
            b[:, 0] = S - b[:, 0] - b[:, 2]
        if torch.rand(1) < 0.5:                        # flip vertical
            img = img.flip(-2)
            b[:, 1] = S - b[:, 1] - b[:, 3]
        if torch.rand(1) < 0.5:                        # rotación de 90° antihoraria
            img = torch.rot90(img, 1, dims=(-2, -1))
            x, y, w, h = b[:, 0].clone(), b[:, 1].clone(), b[:, 2].clone(), b[:, 3].clone()
            b[:, 0], b[:, 1], b[:, 2], b[:, 3] = y, S - x - w, h, w
        return img, b
