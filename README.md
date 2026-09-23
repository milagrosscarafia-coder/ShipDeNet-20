# ShipDeNet-20

Implementación en PyTorch de **ShipDeNet-20** [1], un detector liviano de barcos en imágenes SAR
(20 capas convolucionales, < 1 MB), como proyecto final del curso *Aprendizaje profundo con
aplicación a visión artificial* (Instituto Balseiro).

## Estado

- [x] Lectura del dataset (`rdata.py`) y tests
- [x] Cálculo de anchors con k-means (`anchors.py`) y tests
- [ ] Backbone con DS-Conv (15 capas)
- [ ] Loss (Ec. 22–25 de [2])
- [ ] Post-proceso (decodificación + NMS) y evaluación (AP50)
- [ ] Módulos FF, FE y SSFP
- [ ] Entrenamiento y ablación (Tabla I de [1])


## Dataset

Official-SSDD [3], versión con cajas horizontales en formato COCO:

```
Official-SSDD-OPEN/BBox_SSDD/coco_style/
```

## Estructura

```
read_data_set/
├── rdata.py          # lectura del dataset
├── anchors.py        # cálculo de los 9 anchors
├── test_rdata.py
└── test_anchors.py
```

## read_data_set/rdata.py

**`SSDD_BBox_coco(img_dir, ann_file)`**: dataset de PyTorch que carga las imágenes y sus cajas.

- Redimensiona cada imagen a L × L, con L = 160.
- `img, boxes = dataset[i]` devuelve la imagen y un tensor `boxes` de forma `(n_i, 4)`, con una fila por barco.
- Formato de las cajas: `[x, y, w, h]` en píxeles de la imagen de 160 × 160, con `(x, y)` la esquina
  superior izquierda (formato COCO), escaladas con el mismo factor que la imagen.

**`print_imagen(img, boxes)`**: muestra la imagen con las cajas dibujadas.


## read_data_set/test_rdata.py 

    Testbench (generado con Claude) para verificar que los datos se leen bien.

## anchors.py 


Calcula los 9 anchors (w, h) con k-means sobre las cajas del conjunto de **entrenamiento**, usando la
distancia de [2] (Ec. 16):

- **`def load_train_wh(img_dir, ann_file):`**
    Retorna `wh` que contiene todos los anchos y altos de las cajas que se encuentran en una imagen. 
    `wh = torch.cat(wh, dim=0).float()` Antes de esa línea, wh es una lista con un tensor por imagen. Cada tensor tiene forma (n_i, 2), donde n_i es la cantidad de barcos de la imagen i, ahora retona un \'unico tensor de dimension (N,2) N = total de cajar del dataset.
- `def IoU_wh(boxes, centroides):`
    Calcula la metrica utlizada por kmeans. Como se define en [12], la metrica utilizada es \frac{interseccion},{union} de las areas de dos cajas. en tra funcion se calcula la m\'etrica de cada box con cada centroide de la clasificacion. 
- **`kmeans_iou(boxes, k, seed=0)`**: k-means con la distancia anterior.
  1. Inicializa k centroides con cajas del dataset elegidas al azar (con semilla, para que sea reproducible).
  2. Asigna cada caja al centroide de mayor IoU.
  3. Recalcula cada centroide como la **mediana** de sus cajas. Los clusters vacíos se reinician.
  4. Repite hasta que ninguna caja cambie de cluster, o hasta un máximo de `n_iter` iteraciones.

  Devuelve los centroides ordenados por área (de chico a grande: los 3 primeros van a L/8, los 3
  siguientes a L/16 y los 3 últimos a L/32) y el IoU medio.

- `def best_of_runs(boxes, k=9, runs=10):`
    Corre `kmeans_iou` con `runs` semillas distintas y se queda con el resultado de mayor IoU medio, porque k-means depende de la inicialización.
    
## test_anchors.py
    Testbench generado con claude para anchors.py.
