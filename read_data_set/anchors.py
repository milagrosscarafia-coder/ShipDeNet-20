from read_data_set.rdata import SSDD_BBox_coco
import torch


# recibe la ruta de las imágenes y del json COCO, devuelve
# una lista con el ancho y alto de cada caja, escalado a 160×160. 

def load_train_wh(img_dir, ann_file): 
    dataset = SSDD_BBox_coco(img_dir, ann_file)
    wh = []
    for i in range(len(dataset.images)): 
        _, b = dataset[i]
        wh.append(b[:, 2:4]) # solo guardamos el ancho y alto de cada caja

    wh = torch.cat(wh, dim=0).float()

    return wh[(wh[:, 0] > 0) & (wh[:, 1] > 0)]   # descartar cajas degeneradas


# dimeneiones boxes:(N,2), centroides:(k,2) 
# N = numero de cajas de entrenamiento, k = numero de anchors
# Como estamos usando BBox centradas en el origen, la intersección es el producto 
# de los mínimos de ancho y alto. Ademas, la union es la suma de las areas menos la interseccion. 

def IoU_wh(boxes, centroides):
    inter = torch.min(boxes[:,None,0],centroides[None,:,0]) * torch.min(boxes[:,None,1], centroides[None, :, 1])

    area_b = boxes[:,0] * boxes[:,1]
    area_c = centroides[:,0] * centroides[:,1]
    union = area_b[:,None] + area_c[None,:] - inter
    iou = inter / union
    return iou

# kmeans_iou es el k-means: asigna cada caja al centroide con mayor IoU, recalcula cada 
# centroide como la mediana de sus
# cajas y repite hasta que nada cambie. Al final ordena los anchors de chico a grande.
# best_of_runs corre el k-means 10 veces con distintas inicializaciones y se queda con el
# mejor, porque el resultado depende
# del punto de partida.

def kmeans_iou(boxes, k, seed = 0 ): #modificado

    g = torch.Generator().manual_seed(seed)

    centroides = boxes[torch.randperm(len(boxes), generator=g)[:k]].clone()
    # no se capte la misma caja
    prev = None
    while True:
        iou = IoU_wh(boxes,centroides)
        assign = torch.argmax(iou, dim = 1) # se queda para cada box con el centroide de mayor iou
        for j in range(k):
            members = boxes[assign == j]            # cajas del cluster j, forma (n_j, 2)
            if len(members) > 0:
                centroides[j] = torch.median(members, dim=0).values
            else:
                peor = IoU_wh(boxes, centroides).max(dim=1).values.argmin()   # caja peor cubierta
                centroides[j] = boxes[peor]                                   # cluster vacío: reiniciarlo ahí

        if prev is not None and torch.equal(assign, prev):
            break   
        prev = assign

    area = centroides[:, 0] * centroides[:, 1]
    centroides = centroides[torch.argsort(area)]       # chico -> grande
    mean_iou = IoU_wh(boxes, centroides).max(dim=1).values.mean().item()
    return centroides, mean_iou


def best_of_runs(boxes, k=9, runs=10):
    resultados = [kmeans_iou(boxes, k, seed=s) for s in range(runs)]
    return max(resultados, key=lambda r: r[1])





