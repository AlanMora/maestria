# Webcam filters: edge detection and green-color segmentation with k-means.
# Requirements: pip install opencv-python numpy
import cv2
import numpy as np

FILTER = "segmentation"  # "edges" or "segmentation"

N_COLORS = 10  # color clusters (B, G, R)
N_GREEN = 3    # position clusters (x, y) for green pixels

color_centers = np.random.rand(N_COLORS, 3) * 255
green_centers = np.random.rand(N_GREEN, 2) * [640, 480]


def edges(frame):
    blue = frame[:, :, 0].astype(np.float32)

    # Prewitt masks
    mask_x = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], np.float32)
    mask_y = mask_x.T

    grad_x = cv2.filter2D(blue, -1, mask_x)
    grad_y = cv2.filter2D(blue, -1, mask_y)

    result = np.zeros_like(frame)
    result[grad_x > 30] = (255, 0, 0)  # vertical edges in blue
    result[grad_y > 30] = (0, 0, 255)  # horizontal edges in red
    return result


def segmentation(frame):
    h, w, _ = frame.shape
    pixels = frame.reshape(-1, 3).astype(np.float32)

    # 1. Assign each pixel to its closest color cluster
    distances = np.linalg.norm(pixels[:, None, :] - color_centers[None, :, :], axis=2)
    labels = distances.argmin(axis=1)

    # 2. Find which clusters are "green"
    B, G, R = color_centers[:, 0], color_centers[:, 1], color_centers[:, 2]
    green_clusters = (R > G) & (R > B) & (B < 150) & (G< 150)
    is_green = green_clusters[labels].reshape(h, w)

    # 3. Paint green pixels with their cluster color, everything else black
    cluster_colors = color_centers[labels].reshape(h, w, 3).astype(np.uint8)
    result = np.zeros_like(frame)
    result[is_green] = cluster_colors[is_green]
    frame[is_green] = 0
    frame[is_green, 1] = cluster_colors[is_green, 1]

    # 4. Move each color cluster to the mean of its pixels
    for k in range(N_COLORS):
        members = pixels[labels == k]
        if len(members) > 0:
            color_centers[k] = members.mean(axis=0)

    # 5. Run one k-means step on the (x, y) positions of the green pixels
    ys, xs = np.nonzero(is_green)
    points = np.stack([xs, ys], axis=1)
    if len(points) > 0:
        d = np.linalg.norm(points[:, None, :] - green_centers[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        for k in range(N_GREEN):
            members = points[nearest == k]
            if len(members) > 0:
                green_centers[k] = members.mean(axis=0)

    # 6. Draw an X on each green cluster
    for x, y in green_centers.astype(int):
        cv2.line(result, (x - 30, y + 30), (x + 30, y - 30), (255, 255, 0))
        cv2.line(result, (x + 30, y + 30), (x - 30, y - 30), (255, 255, 0))

    return result


def main():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open the camera")
        return

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if FILTER == "edges":
            result = edges(frame)
        else:
            result = segmentation(frame)

        cv2.imshow("camera", frame)
        cv2.imshow("salida", result)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # q or Esc
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
