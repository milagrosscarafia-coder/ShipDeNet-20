"""Tests de anchors.py (k-means con d = 1 - IoU sobre el (w, h) de las cajas).

Correr desde la raiz del repo:
    python -m pytest tests/test_anchors.py -v
"""
import os, sys, threading

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "read_data_set"))
import anchors
from anchors import IoU_wh, kmeans_iou, best_of_runs, load_train_wh

TIMEOUT = 5  # segundos


def run_with_timeout(fn, *args, timeout=TIMEOUT, **kwargs):
    """Corre fn en un thread; falla si no termina a tiempo y re-lanza sus excepciones."""
    out = {}

    def target():
        try:
            out["res"] = fn(*args, **kwargs)
        except BaseException as e:
            out["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"{fn.__name__} no termino en {timeout}s (loop infinito?)")
    if "err" in out:
        raise out["err"]
    return out["res"]


@pytest.fixture
def random_boxes():
    g = torch.Generator().manual_seed(123)
    return 1 + 60 * torch.rand(300, 2, generator=g)   # (w, h) en [1, 61)


@pytest.fixture
def two_clusters():
    return torch.tensor([[10.0, 10.0], [9.5, 10.5], [10.5, 9.5],
                         [10.0, 40.0], [9.5, 41.0], [10.5, 39.0]])


# --------------------------------------------------------------------- IoU_wh

class TestIoU:
    def test_shape(self, random_boxes):
        c = random_boxes[:7]
        assert IoU_wh(random_boxes, c).shape == (300, 7)

    def test_identical_boxes(self, random_boxes):
        iou = IoU_wh(random_boxes, random_boxes)
        assert torch.allclose(iou.diagonal(), torch.ones(300))

    def test_known_value(self):
        iou = IoU_wh(torch.tensor([[10.0, 20.0]]), torch.tensor([[10.0, 10.0]]))
        assert iou.item() == 0.5

    def test_symmetry(self, random_boxes):
        a, b = random_boxes[:50], random_boxes[50:80]
        assert torch.allclose(IoU_wh(a, b), IoU_wh(b, a).T)

    @pytest.mark.parametrize("s", [0.1, 3.0, 160.0])
    def test_scale_invariance(self, random_boxes, s):
        a, b = random_boxes, random_boxes[:9]
        assert torch.allclose(IoU_wh(a, b), IoU_wh(a * s, b * s), atol=1e-6)

    def test_range(self, random_boxes):
        iou = IoU_wh(random_boxes, random_boxes[:20])
        assert (iou > 0).all() and (iou <= 1 + 1e-6).all()


# ----------------------------------------------------------------- kmeans_iou

class TestKmeans:
    def test_recovers_two_clusters(self, two_clusters):
        c, miou = run_with_timeout(kmeans_iou, two_clusters, 2)
        assert torch.allclose(c, torch.tensor([[10.0, 10.0], [10.0, 40.0]]), atol=1.0)
        assert miou > 0.9

    def test_shape_and_sorted_by_area(self, random_boxes):
        c, _ = run_with_timeout(kmeans_iou, random_boxes, 9)
        assert c.shape == (9, 2)
        area = c[:, 0] * c[:, 1]
        assert (area[1:] >= area[:-1]).all()

    def test_reproducible(self, random_boxes):
        c1, m1 = run_with_timeout(kmeans_iou, random_boxes, 9, seed=7)
        c2, m2 = run_with_timeout(kmeans_iou, random_boxes, 9, seed=7)
        assert torch.equal(c1, c2) and m1 == m2

    @pytest.mark.parametrize("seed", range(5))
    def test_k_equal_distinct_boxes_gives_iou_1(self, seed):
        # 4 tamaños distintos, cada uno repetido 3 veces
        distintas = torch.tensor([[5.0, 5.0], [8.0, 20.0], [30.0, 12.0], [40.0, 40.0]])
        boxes = distintas.repeat_interleave(3, dim=0)
        c, miou = run_with_timeout(kmeans_iou, boxes, 4, seed=seed)
        assert miou == pytest.approx(1.0)
        assert torch.equal(c, distintas)   # distintas ya esta ordenada por area

    def test_empty_cluster_branch(self):
        boxes = torch.tensor([[12.0, 12.0]]).repeat(50, 1)
        c, miou = run_with_timeout(kmeans_iou, boxes, 3)
        assert c.shape == (3, 2)
        assert torch.isfinite(c).all()
        assert miou == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(5))
    def test_terminates(self, random_boxes, seed):
        c, miou = run_with_timeout(kmeans_iou, random_boxes, 9, seed=seed)
        assert torch.isfinite(c).all() and 0 < miou <= 1


# --------------------------------------------------------------- best_of_runs

def test_best_of_runs_is_best(random_boxes):
    runs = 5
    _, best = run_with_timeout(best_of_runs, random_boxes, k=4, runs=runs)
    for s in range(runs):
        _, m = run_with_timeout(kmeans_iou, random_boxes, 4, seed=s)
        assert best >= m


# -------------------------------------------------------------- load_train_wh

# cajas COCO (x, y, w, h) ya escaladas, tal como las devuelve SSDD_BBox_coco[i]
MOCK_BOXES = [
    [[1.0, 2.0, 10.0, 20.0], [5.0, 5.0, 3.0, 4.0]],
    [],                                               # imagen sin barcos
    [[0.0, 0.0, 0.0, 15.0], [7.0, 8.0, 12.5, 6.0]],   # la primera tiene w = 0
]
EXPECTED_WH = torch.tensor([[10.0, 20.0], [3.0, 4.0], [12.5, 6.0]])


class FakeSSDD:
    def __init__(self, img_dir, ann_file, size=160):
        self.images = [{"id": i} for i in range(len(MOCK_BOXES))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        b = torch.tensor(MOCK_BOXES[i], dtype=torch.float32).reshape(-1, 4)
        return torch.zeros(1, 160, 160), b


def test_load_train_wh(monkeypatch):
    monkeypatch.setattr(anchors, "SSDD_BBox_coco", FakeSSDD)
    wh = load_train_wh("no/importa", "no/importa.json")
    assert isinstance(wh, torch.Tensor) and wh.dtype == torch.float32
    assert wh.shape == (3, 2)                  # la caja con w = 0 se descarta
    assert (wh > 0).all()
    assert torch.equal(wh, EXPECTED_WH)
