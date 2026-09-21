**ShipDeNet-20**
DataSet: Official-SSDD-OPEN/BBox_SSDD/coco_style

## read_data_set/rdata.py

Cargar imagenes y crear boxes. 
Redimensionar LxL L = 160. 
- Boxes = `[x, y, w, h]`, con `(x, y)` la esquina superior izquierda,
escaladas con el mismo factor que la imagen.
`print_imagen(img, boxes)` — muestra la imagen con las cajas dibujadas.

## read_data_set/test_rdata.py

Testbench (generado con Claude) para verificar que los datos se leen bien.
