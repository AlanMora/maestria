"""Interactive MLP demo (Python/tkinter port of MainPage.xaml.cpp).

Pick a class, click on the canvas to add samples, press Train, then Evaluate.
"""
import tkinter as tk
from tkinter import ttk

from neural_network import build_network

WIDTH, HEIGHT = 700, 500
EVAL_STEP = 4  # pixel block size used when rendering the decision map
NUM_CLASSES = 5
CLASS_COLORS = ["red", "yellow", "green", "cyan", "blue"]


def hue_to_rgb(p, q, t):
    if t < 0.0:
        t += 1.0
    if t > 1.0:
        t -= 1.0
    if t < 1.0 / 6.0:
        return p + (q - p) * 6.0 * t
    if t < 0.5:
        return q
    if t < 2.0 / 3.0:
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0
    return p


def hsl_to_rgb(h, s, l):
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1.0 / 3.0)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1.0 / 3.0)
    return int(r * 255), int(g * 255), int(b * 255)


def to_net(px, py):
    return 2.0 * px / WIDTH - 1.0, 2.0 * py / HEIGHT - 1.0


class App:
    def __init__(self, root):
        self.root = root
        root.title("MLP - Homework")
        self.nn = build_network()
        self.samples = []  # (x_px, y_px, class_index)

        pane = tk.Frame(root, padx=6, pady=6)
        pane.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(pane, text="Evaluate", command=self.evaluate).pack(fill=tk.X)
        self.progress = ttk.Progressbar(pane, maximum=100, length=100)
        self.progress.pack(fill=tk.X, pady=(2, 8))

        tk.Button(pane, text="Train", command=self.train).pack(fill=tk.X)
        tk.Label(pane, text="Epochs").pack(anchor=tk.W)
        self.epochs = tk.IntVar(value=1000)
        tk.Entry(pane, textvariable=self.epochs, width=8).pack(fill=tk.X, pady=(0, 8))

        self.selected = tk.IntVar(value=0)
        for c in range(NUM_CLASSES):
            tk.Radiobutton(pane, text=f"C{c + 1}", variable=self.selected, value=c,
                           fg=CLASS_COLORS[c] if c != 1 else "goldenrod").pack(anchor=tk.W)

        tk.Button(pane, text="Reset", command=self.reset).pack(fill=tk.X, pady=(8, 0))
        self.status = tk.Label(pane, text="", justify=tk.LEFT, wraplength=110)
        self.status.pack(anchor=tk.W, pady=(8, 0))

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_click)
        self.image = None

    def on_click(self, event):
        c = self.selected.get()
        self.samples.append((event.x, event.y, c))
        self.draw_marker(event.x, event.y, c)
        self.status.config(text=f"Samples: {len(self.samples)}")

    def draw_marker(self, x, y, c):
        color = CLASS_COLORS[c]
        self.canvas.create_line(x - 3, y - 3, x + 3, y + 3, fill=color, width=2, tags="marker")
        self.canvas.create_line(x + 3, y - 3, x - 3, y + 3, fill=color, width=2, tags="marker")

    def train(self):
        if not self.samples:
            return
        data = []
        for x, y, c in self.samples:
            d = [0.0] * NUM_CLASSES
            d[c] = 1.0
            data.append((to_net(x, y), d))

        epochs = max(1, self.epochs.get())
        err = 0.0
        for j in range(epochs):
            err = 0.0
            for inputs, d in data:
                err += abs(self.nn.train(inputs, d))
            if j % 20 == 0:
                self.progress["value"] = 100 * j / epochs
                self.status.config(text=f"Epoch {j}\nError {err:.3f}")
                self.root.update()
        self.progress["value"] = 100
        self.status.config(text=f"Trained {epochs} epochs\nError {err:.3f}")

    def evaluate(self):
        w, h = WIDTH // EVAL_STEP, HEIGHT // EVAL_STEP
        img = tk.PhotoImage(width=w, height=h)
        rows = []
        for n in range(h):
            row = []
            for m in range(w):
                x, y = to_net(m * EVAL_STEP, n * EVAL_STEP)
                sel, conf = self.nn.evaluate((x, y))
                r, g, b = hsl_to_rgb(sel / 6.0, 0.9, conf / 2.0)
                row.append(f"#{r:02x}{g:02x}{b:02x}")
            rows.append("{" + " ".join(row) + "}")
            if n % 10 == 0:
                self.progress["value"] = 100 * n / h
                self.root.update()
        img.put(" ".join(rows))
        self.image = img.zoom(EVAL_STEP, EVAL_STEP)
        self.canvas.delete("map")
        self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW, tags="map")
        self.canvas.tag_raise("marker")
        self.progress["value"] = 100

    def reset(self):
        self.nn = build_network()
        self.samples.clear()
        self.canvas.delete("all")
        self.progress["value"] = 0
        self.status.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
