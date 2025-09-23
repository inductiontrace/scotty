import numpy as np


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1 + 1.0)
    ih = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0))
    area_b = max(0.0, (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / float(denom)


def simple_nms(dets, iou_thr=0.6):
    """
    dets: list of ((x1,y1,x2,y2), conf, cls)
    class-agnostic NMS to collapse duplicates in the *same frame*.
    """
    if not dets:
        return dets
    boxes = np.array([d[0] for d in dets], dtype=float)
    scores = np.array([d[1] for d in dets], dtype=float)
    keep = []
    idxs = scores.argsort()[::-1]
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        rest = idxs[1:]
        if rest.size == 0:
            break
        suppressed = {
            int(j) for j in rest if iou_xyxy(boxes[i], boxes[j]) > iou_thr
        }
        if suppressed:
            rest = np.array([int(k) for k in rest if int(k) not in suppressed], dtype=int)
        idxs = rest
    return [dets[i] for i in keep]
