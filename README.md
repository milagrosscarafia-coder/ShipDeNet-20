# ShipDeNet-20

Implementación en PyTorch de **ShipDeNet-20** [1], un detector liviano de barcos en imágenes SAR
(20 capas convolucionales, < 1 MB), como proyecto final del curso *Aprendizaje profundo con
aplicación a visión artificial* (Instituto Balseiro).

## Estado

- [x] Lectura del dataset (`rdata.py`) y tests
- [x] Cálculo de anchors con k-means (`anchors.py`) y tests
- [x] Backbone con DS-Conv (15 capas)
- [x] Loss (Ec. 22–25 de [2]) (implementada por claude)
- [x] Post-proceso (decodificación + NMS) y evaluación (AP50)
- [x] Módulos FF, FE y SSFP
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

## models/

### layers.py

- **`ConvAC`**: convolución convencional + BatchNorm + Leaky-ReLU. Se usa en la capa 1,
  donde la imagen tiene un solo canal. Con `head=True` devuelve la salida lineal, con bias.
- **`DSConv`**: convolución separable en profundidad (DS-Conv), el bloque básico de la red:
  - *depthwise*: un filtro espacial por canal, cada canal por separado (y el downsampling si `stride > 1`);
  - *pointwise*: convolución 1×1 que mezcla los canales y fija la cantidad de salida.

  Cada una va seguida de BatchNorm + Leaky-ReLU (Fig. 5b de [2]). Con `head=True`, la pointwise
  final es lineal y con bias, para usarla como capa de salida.

### backbone.py

- **`Backbone`**: las 15 capas de la Fig. 1(a) de [1]. Reduce la imagen de L a L/32 y devuelve las
  salidas intermedias que usan los módulos: la capa 1 (L/2), la capa 3 (L/4), y las listas de
  capas por resolución: 4-6 (L/8), 7-10 (L/16) y 11-15 (L/32).

### modules.py

Los tres módulos que se agregan después del backbone:

- **`FF`** (*feature fusion*): capas 16 y 17. Llevan características superficiales a la
  resolución de las profundas, para poder fusionarlas:
  - capa 16: capa 1 (L/2) → L/32, con kernel 17×17 y stride 16;
  - capa 17: capa 3 (L/4) → L/16, con kernel 5×5 y stride 4.
- **`FE`** (*feature enhance*): concatena por canales todas las capas de la misma resolución:
  11-15 + 16 en L/32, 7-10 + 17 en L/16, y 4-6 en L/8. No tiene parámetros.
- **`SSFP`** (*scale share feature pyramid*): capas 18, 19 y 20, una capa de detección por escala.
  Con `share=True`, suma las salidas entre escalas bajando o subiendo su resolución
  (Ec. 1-3 de [1]); con `share=False`, cada escala predice por separado.

## loss.py 
- **`to_cxcywh_norm`**: convierte las cajas [x, y, w, h]  en pixeles (esquina sup. izq., cocos) -> a (cx, cy, w, h)con coordenadas en el centros, normalizado a [0, 1]."""

- **`shape_iou`**: Realiza el IoU entre una caja (w, h) y cada anchor (K, 2), todas centradas en el origen. Solo compara los tamanos es la elegir el anchor de cada barco. 

- **`box_iou`**: Realiza IoU entre barcos con la posicion real en formato (cx, cy, w, h), se usa como target del score. 

- **`build_targets`** Traduce las cajas del ground truth a tensores con la forma de la salida de la red. Para cada barco 
    1. Elige entre los 9 anchors el de forma mas parecia con `shape_iou`.
    2. Busca la celda de esa escala que contiene el centro del barco
    3. Escribe los targets en esa posici\'on `[imagen, anchor, fila, columna]`

Devuelve, para cada escala (orden L/32, L/16, L/8):
| Tensor | Forma | Contenido |
|---|---|---|
| `obj` | (B, 3, S, S) | P_cell(ship): 1 donde hay un barco asignado (Ec. 13) |
| `txy` | (B, 3, S, S, 2) | offset del centro dentro de la celda, en [0, 1] (Ec. 22) |
| `box` | (B, 3, S, S, 4) | caja real `(cx, cy, w, h)` normalizada (Ec. 23 y el IoU) |
[B es la imagen del batch, 3 hay 3 tipos de anchores, S,S la fila y columna de la celda (S=5,10,20 dependiendo la escala), con estas dimensiones se arma la grilla, `obj` recibe un solo numero por casillero de grilla, `txy` recibe 2 y `box` recibe 4]

- **`ShipDeNetLoss(nn.Module)`**

    Primero: Decodifica la salidad de la red 
    - (B, 3*VALS, S, S)-->(B, 3, S, S, VALS)
    - Centros: sigmoide, para que no se corran de la celda correspondiente
    - tamaño: anchor · e^t, una corrección multiplicativa del anchor (siempre positiva, $$w = \text{anchor}_w \cdot e^{t_w} \qquad h = \text{anchor}_h \cdot e^{t_h}$$);
    - score: sigmoide, en [0, 1].

    Después calcula los términos de la loss, solo donde `obj = 1` salvo el último:

    - **L_xy** (Ec. 22): error cuadrático del centro;
    - **L_wh** (Ec. 23): error cuadrático entre las raíces de w y h;
    - **L_obj** (Ec. 24): donde hay barco, el score debe valer el IoU de su caja con la real;
    - **L_noobj** (Ec. 24): donde no hay barco, el score debe valer 0.


  Total (Ec. 25), con α = β = 5 y γ = 0.5:

    loss = α·L_xy + β·L_wh + γ·L_score,  con  L_score = (1/γ)·L_obj + L_noobj 
    = α·L_xy + β·L_wh + L_obj + γ·L_noobj

