import cv2
import numpy as np

def fit_canvas(image, size):
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('expected nonempty BGR image')
    h, w = image.shape[:2]
    factor = min(size / h, size / w)
    nh = min(size, max(1, round(h * factor)))
    nw = min(size, max(1, round(w * factor)))
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]], axis=0)
    background = np.median(border, axis=0).astype(image.dtype)
    canvas = np.empty((size, size, 3), dtype=image.dtype)
    canvas[...] = background
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    mask = np.zeros((1, size, size), dtype=np.float32)
    mask[:, y:y + nh, x:x + nw] = 1
    return canvas, mask, {'source_h': h, 'source_w': w, 'resized_h': nh,
                          'resized_w': nw, 'x': x, 'y': y,
                          'valid_area_fraction': float(mask.mean())}


