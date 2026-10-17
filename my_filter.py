# my_filter5.py - Python version of My_Filter5.cpp
# Requirements: pip install opencv-python numpy
import cv2
import numpy as np

# 5x5 Gaussian blur mask: rows of Pascal's triangle (1 4 6 4 1) multiplied together

BLUR_MASK = np.array([[1, 4, 6, 4, 1],
                      [4, 16, 24, 16, 4],
                      [6, 24, 36, 24, 6],
                      [4, 16, 24, 16, 4],
                      [1, 4, 6, 4, 1]]) / 256.0
"""

BLUR_MASK = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]]) / 16.0
"""
# 3x3 Sobel mask (detects vertical edges)
#SOBEL_MASK = np.array([[-1, 0, 1],
#                       [-2, 0, 2],
#                       [-1, 0, 1]])

SOBEL_MASK = np.array([[0, 9, 0],
                       [9, -36, 9],
                       [0, 9, 0]])


def to_gray(frame):
    """Intensity I = (R + G + B) / 3 for every pixel."""
    return frame.astype(np.float32).mean(axis=2)


def convolve(gray, mask):
    """Slide the mask over the image: out(j,i) = sum mask[m,n] * gray(j+m, i+n)."""
    k = mask.shape[0]
    h, w = gray.shape
    out = np.zeros((h - k + 1, w - k + 1), np.float32)
    for m in range(k):
        for n in range(k):
            out += mask[m, n] * gray[m:m + h - k + 1, n:n + w - k + 1]
    return out


def my_blur(frame):
    """Blur the image with the 5x5 Gaussian mask."""
    blurred = convolve(to_gray(frame), BLUR_MASK)
    return np.pad(blurred, 2).astype(np.uint8)


def my_borde2(gray):
    """Sobel edge detection followed by a threshold (white = edge)."""
    edges = convolve(gray.astype(np.float32), SOBEL_MASK)
    binary = np.where(edges > 128, 255, 0)
    return np.pad(binary, 1).astype(np.uint8)


# ---- Color segmentation (k-means style), not used by default ----
NUM_CLUSTERS = 10
NUM_POINTS = 2
clusters = np.random.randint(0, 256, (NUM_CLUSTERS, 3)).astype(np.float32)  # B, G, R
points = np.random.randint(0, [640, 480], (NUM_POINTS, 2)).astype(np.float32)  # X, Y


def my_segmentation(frame):
    """Paint 'red' pixels with their cluster color, the rest in gray, and track red blobs."""
    global clusters, points
    pixels = frame.reshape(-1, 3).astype(np.float32)

    # 1. Assign each pixel to the nearest color cluster (Manhattan distance)
    dist = np.abs(pixels[:, None, :] - clusters[None, :, :]).sum(axis=2)
    label = dist.argmin(axis=1)

    # 2. A cluster is "red" if R > 170 and G, B < 170
    b, g, r = clusters[:, 0], clusters[:, 1], clusters[:, 2]
    is_red = ((r > 170) & (b < 170) & (g < 170))[label]

    gray = pixels.mean(axis=1, keepdims=True).repeat(3, axis=1)
    result = np.where(is_red[:, None], clusters[label], gray)
    result = result.reshape(frame.shape).astype(np.uint8)

    # 3. Update cluster colors with the mean of their pixels
    for k in range(NUM_CLUSTERS):
        if np.any(label == k):
            clusters[k] = pixels[label == k].mean(axis=0)

    # 4. Group red pixels into NUM_POINTS positions and move each one to its mean
    ys, xs = np.unravel_index(np.nonzero(is_red)[0], frame.shape[:2])
    if len(xs) > 0:
        xy = np.stack([xs, ys], axis=1).astype(np.float32)
        near = np.abs(xy[:, None, :] - points[None, :, :]).sum(axis=2).argmin(axis=1)
        for l in range(NUM_POINTS):
            if np.any(near == l):
                points[l] = xy[near == l].mean(axis=0)

    for x, y in points.astype(int):
        cv2.rectangle(result, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0))
    return result


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("fail!")
        return

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        # result = my_segmentation(frame)
        blurred = my_blur(frame)
        edges = my_borde2(blurred)

        cv2.imshow("camera", frame)
        cv2.imshow("salida", edges)
        #cv2.imshow("salida", blurred)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # q or ESC to quit
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()