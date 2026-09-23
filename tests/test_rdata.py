"""Testbench for rdata.py — checks images and labels are read correctly.

Usage:
    python test_rdata.py          # tests the test split
    python test_rdata.py train    # tests the train split
"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")          # sin ventanas: los graficos se guardan a PNG
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdata import SSDD_BBox_coco, print_imagen

ROOT = (r"C:\Users\Usuario\Desktop\5to Semestre\Aprendizaje profundo con vision"
        r" artificial\ShipDeNet\Official-SSDD-OPEN\Official-SSDD-OPEN"
        r"\BBox_SSDD\coco_style")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
IMG_DIR = os.path.join(ROOT, "images", SPLIT)
ANN_FILE = os.path.join(ROOT, "annotations", f"{SPLIT}.json")
SIZE = 160
TOL = 1e-3

passed = failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"\n         -> {detail}" if detail else ""))
    return ok


print(f"\n{'='*70}\n TESTBENCH rdata.py  --  split: {SPLIT}\n{'='*70}")

# ---------------------------------------------------------------- 1. rutas
print("\n[1] Rutas y archivos")
ok_dir = check("la carpeta de imagenes existe", os.path.isdir(IMG_DIR), IMG_DIR)
ok_ann = check("el archivo de anotaciones existe", os.path.isfile(ANN_FILE), ANN_FILE)
if not (ok_dir and ok_ann):
    print("\nNo se puede continuar sin los datos. Revisa las rutas.")
    sys.exit(1)

coco = json.load(open(ANN_FILE))
check("el json tiene 'images' y 'annotations'",
      "images" in coco and "annotations" in coco, f"claves: {list(coco.keys())}")

n_img_json = len(coco["images"])
n_ann_json = len(coco["annotations"])
print(f"         json: {n_img_json} imagenes, {n_ann_json} cajas")

# ------------------------------------------------- 2. construccion dataset
print("\n[2] Construccion del dataset")
data = SSDD_BBox_coco(img_dir=IMG_DIR, ann_file=ANN_FILE, size=SIZE)

check("len(dataset) == numero de imagenes del json",
      len(data) == n_img_json, f"{len(data)} != {n_img_json}")

n_boxes_stored = sum(len(v) for v in data.boxes.values())
check("se guardaron TODAS las cajas del json",
      n_boxes_stored == n_ann_json,
      f"guardadas {n_boxes_stored}, esperadas {n_ann_json}")

check("hay una entrada en self.boxes por cada imagen",
      len(data.boxes) == n_img_json, f"{len(data.boxes)} != {n_img_json}")

faltantes = [im["file_name"] for im in coco["images"]
             if not os.path.isfile(os.path.join(IMG_DIR, im["file_name"]))]
check("todos los archivos de imagen del json existen en disco",
      not faltantes, f"{len(faltantes)} faltan, ej: {faltantes[:3]}")

# ------------------------------- 3. el json describe bien las imagenes reales
print("\n[3] Coherencia json <-> archivo real (muestra de 20)")
malas = []
for im in coco["images"][:20]:
    with Image.open(os.path.join(IMG_DIR, im["file_name"])) as f:
        if f.size != (im["width"], im["height"]):
            malas.append((im["file_name"], f.size, (im["width"], im["height"])))
check("width/height del json coinciden con el archivo real",
      not malas, f"{malas[:3]}")

# ------------------------------------------- 4. recorrido de TODAS las muestras
print(f"\n[4] Recorriendo las {len(data)} muestras")
err_shape, err_range, err_box_shape, err_bounds, err_wh, err_scale, err_crash = \
    [], [], [], [], [], [], []
total_boxes = sin_barcos = 0
min_px, max_px = float("inf"), float("-inf")

# indice rapido: image_id -> cajas crudas del json
crudas = {im["id"]: [] for im in coco["images"]}
for a in coco["annotations"]:
    crudas[a["image_id"]].append(a["bbox"])

for i in range(len(data)):
    info = coco["images"][i]
    try:
        img, b = data[i]
    except Exception as e:
        err_crash.append((i, info["file_name"], repr(e)))
        continue

    # --- imagen
    if tuple(img.shape) != (1, SIZE, SIZE):
        err_shape.append((i, tuple(img.shape)))
    if img.dtype != torch.float32 or img.min() < -TOL or img.max() > 1 + TOL:
        err_range.append((i, img.dtype, float(img.min()), float(img.max())))
    min_px, max_px = min(min_px, float(img.min())), max(max_px, float(img.max()))

    # --- cajas
    if b.ndim != 2 or b.shape[1] != 4 or b.dtype != torch.float32:
        err_box_shape.append((i, tuple(b.shape), b.dtype))
        continue

    total_boxes += b.shape[0]
    if b.shape[0] == 0:
        sin_barcos += 1
        continue

    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if (w <= 0).any() or (h <= 0).any():
        err_wh.append((i, info["file_name"]))
    if (x < -TOL).any() or (y < -TOL).any() or \
       (x + w > SIZE + TOL).any() or (y + h > SIZE + TOL).any():
        err_bounds.append((i, info["file_name"],
                           [round(float(v), 1) for v in b.flatten()[:4]]))

    # escalado recalculado de forma independiente desde el json crudo
    sx, sy = SIZE / info["width"], SIZE / info["height"]
    esperado = torch.tensor(crudas[info["id"]], dtype=torch.float32).reshape(-1, 4)
    esperado[:, [0, 2]] *= sx
    esperado[:, [1, 3]] *= sy
    if not torch.allclose(b, esperado, atol=1e-3):
        err_scale.append((i, info["file_name"]))

check("ninguna muestra lanza excepcion", not err_crash, f"{err_crash[:2]}")
check(f"toda imagen es un tensor (1,{SIZE},{SIZE})", not err_shape, f"{err_shape[:3]}")
check("pixeles float32 en rango [0,1]", not err_range, f"{err_range[:3]}")
check("toda caja es un tensor (N,4) float32", not err_box_shape, f"{err_box_shape[:3]}")
check("todo ancho/alto es positivo", not err_wh, f"{err_wh[:3]}")
check(f"toda caja cae dentro de la imagen de {SIZE}px",
      not err_bounds, f"{len(err_bounds)} fuera de rango, ej: {err_bounds[:2]}")
check("el escalado coincide con el recalculo independiente",
      not err_scale, f"{len(err_scale)} mal escaladas, ej: {err_scale[:2]}")
check("el total de cajas devueltas == cajas del json",
      total_boxes == n_ann_json, f"{total_boxes} != {n_ann_json}")

print(f"         rango de pixeles observado: [{min_px:.3f}, {max_px:.3f}]")
print(f"         imagenes sin barcos: {sin_barcos} / {len(data)}")

# ---------------------------------------------------- 5. print_imagen no rompe
print("\n[5] Funcion print_imagen")
con_cajas = next((i for i in range(len(data)) if data[i][1].shape[0] > 0), None)
try:
    img, b = data[con_cajas]
    print_imagen(img, b)
    plt.close("all")
    ok = True
    det = ""
except Exception as e:
    ok, det = False, repr(e)
check("print_imagen(img, b) funciona con cajas", ok, det)

sin = next((i for i in range(len(data)) if data[i][1].shape[0] == 0), None)
if sin is None:
    print("  [skip] no hay imagenes sin barcos en este split")
else:
    try:
        img, b = data[sin]
        print_imagen(img, b)
        plt.close("all")
        ok, det = True, ""
    except Exception as e:
        ok, det = False, repr(e)
    check("print_imagen funciona con 0 cajas", ok, det)

try:
    print_imagen(data[0][0])
    plt.close("all")
    ok, det = True, ""
except Exception as e:
    ok, det = False, repr(e)
check("print_imagen(img) sin cajas (b=None) funciona", ok, det)

# ------------------------------------------------------- 6. control visual
print("\n[6] Control visual")
orden = sorted(range(len(data)), key=lambda i: -len(data.boxes[coco["images"][i]["id"]]))
muestras = orden[:9]

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, i in zip(axes.flat, muestras):
    img, b = data[i]
    ax.imshow(img.squeeze(), cmap="gray")
    for x, y, w, h in b:
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5,
                                       edgecolor="r", facecolor="none"))
    ax.set_title(f"[{i}] {coco['images'][i]['file_name']} - {b.shape[0]} barcos",
                 fontsize=8)
    ax.axis("off")
fig.suptitle(f"SSDD {SPLIT} - verificacion visual de cajas")
fig.tight_layout()
salida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"check_{SPLIT}.png")
fig.savefig(salida, dpi=110)
plt.close(fig)
print(f"  Grilla guardada en: {salida}")
print("  Abrila y confirma que cada rectangulo rojo encierra un barco.")

# --------------------------------------------------------------- resumen
print(f"\n{'='*70}")
print(f" RESULTADO: {passed} passed, {failed} failed")
print(f"{'='*70}\n")
sys.exit(1 if failed else 0)
