import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

user_params = {}


def ask_user_input():
    def browse_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if path:
            entry_file.delete(0, tk.END)
            entry_file.insert(0, path)

    def submit():
        try:
            image_file = entry_file.get()
            w = int(entry_width.get())
            h = int(entry_height.get())
            points = int(entry_points.get())
            user_params['IMAGE_FILE'] = image_file
            user_params['IMAGE_SIZE'] = (w, h)
            user_params['POINT_NUMBER'] = points
            root.destroy()
        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    root = tk.Tk()
    root.title("輸入圖片參數")
    root.geometry("400x300")

    tk.Label(root, text="image_patb:").pack()
    entry_file = tk.Entry(root, width=40)
    entry_file.insert(0, "hushih.jpg")
    entry_file.pack()
    tk.Button(root, text="browse...", command=browse_image).pack()

    tk.Label(root, text="image_widtn:").pack()
    entry_width = tk.Entry(root)
    entry_width.insert(0, "300")
    entry_width.pack()

    tk.Label(root, text="image_height:").pack()
    entry_height = tk.Entry(root)
    entry_height.insert(0, "300")
    entry_height.pack()

    tk.Label(root, text="pin numbers:").pack()
    entry_points = tk.Entry(root)
    entry_points.insert(0, "200")
    entry_points.pack()

    tk.Button(root, text="start generating", command=submit).pack(pady=10)
    root.mainloop()

ask_user_input()

IMAGE_FILE = user_params['IMAGE_FILE']
IMAGE_SIZE = user_params['IMAGE_SIZE']
POINT_NUMBER = user_params['POINT_NUMBER']

LINE_PIXEL = {}
NOT_ALLOWED = set()
USED_LINES = set()
MAX_ERROR = 1e3
Y, X = np.ogrid[:IMAGE_SIZE[1], :IMAGE_SIZE[0]]
center = (IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2)
radius = IMAGE_SIZE[0] // 2
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
MASK = dist_from_center <= radius
drawed = np.zeros(IMAGE_SIZE)

im = Image.open(IMAGE_FILE)
new_image = im.convert(mode="L")
width, height = new_image.size
if width > height:
    top = 0
    bottom = height
    left = (width - height) // 2
    right = left + height
else:
    left = 0
    right = width
    top = (height - width) // 2
    bottom = top + width
new_image = new_image.crop((left, top, right, bottom))
new_image = new_image.resize(IMAGE_SIZE)

pixel_value = np.array(new_image)
pixel_value = 255 - pixel_value
pixel_value[~MASK] = 0

def merge_pixel(pixel1, pixel2):
    return np.maximum(pixel1, pixel2)

def cal_error(pixel1, target_pixel, drawed):
    missing = (target_pixel - pixel1).clip(min=0)
    extra = (pixel1 - target_pixel).clip(min=0)
    if drawed is not None:
        overlap_penalty = (drawed > 200).astype(float)
        extra *= (1 + overlap_penalty * 2)
    return np.mean(4 * missing + 0.5 * extra)

def threshold(distance):
    if distance <= 0.1:
        return 30 * 5
    elif distance <= 0.6:
        return (36 - 60 * distance) * 5
    else:
        return 0

def distanceToLine(slope, y_intersect, i, j):
    i = i - (IMAGE_SIZE[0] / 2)
    j = j - (IMAGE_SIZE[1] / 2)
    if slope == "yee":
        return abs((i - y_intersect))
    return abs(slope * i + y_intersect - j) / math.sqrt(slope ** 2 + 1)

def bresenham_line(x0, y0, x1, y1):
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def pixelOfLine_bresenham(start_index, end_index):
    if start_index == end_index:
        return np.zeros(IMAGE_SIZE, dtype=np.float32)
    angle0 = 2 * math.pi * start_index / POINT_NUMBER
    angle1 = 2 * math.pi * end_index / POINT_NUMBER
    x0 = center[0] + radius * math.cos(angle0)
    y0 = center[1] + radius * math.sin(angle0)
    x1 = center[0] + radius * math.cos(angle1)
    y1 = center[1] + radius * math.sin(angle1)
    coords = bresenham_line(x0, y0, x1, y1)
    pixel_mask = np.zeros(IMAGE_SIZE, dtype=np.float32)
    for x, y in coords:
        if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1] and MASK[y, x]:
            pixel_mask[y, x] = 150
    return pixel_mask

def nextPoint(start, present_pixel, target_pixel):
    min_err = MAX_ERROR
    best_point = None
    if start is None:
        another_point = None
        for i in range(POINT_NUMBER):
            for j in range(i + 1, POINT_NUMBER):
                if ((i, j)) in NOT_ALLOWED or (i, j) in USED_LINES:
                    continue
                line_pixel = LINE_PIXEL[(i, j)]
                line_pixel = merge_pixel(line_pixel, present_pixel)
                err = cal_error(line_pixel, target_pixel, present_pixel)
                if err < min_err:
                    best_point = i
                    another_point = j
                    min_err = err
        return best_point, another_point, min_err
    else:
        for point in range(POINT_NUMBER):
            if point == start or (min(start, point), max(start, point)) in NOT_ALLOWED or (min(start, point), max(start, point)) in USED_LINES:
                continue
            line_pixel = LINE_PIXEL[(min(start, point), max(start, point))]
            line_pixel = merge_pixel(line_pixel, present_pixel)
            err = cal_error(line_pixel, target_pixel, present_pixel)
            if err < min_err:
                best_point = point
                min_err = err
    return best_point, min_err

def process_line(i_j):
    i, j = i_j
    line_pixel = pixelOfLine_bresenham(i, j)
    if np.sum(line_pixel) < 20000:
        NOT_ALLOWED.add((i, j))
    return (i, j), line_pixel

with ThreadPoolExecutor() as executor:
    results = executor.map(process_line, combinations(range(POINT_NUMBER), 2))
for (i, j), pixel in results:
    if (i, j) in NOT_ALLOWED:
        continue
    LINE_PIXEL[(i, j)] = pixel

frames = []
N = 5
fig, ax = plt.subplots()
for i in range(500):
    if i % (N * 10) == 0:
        start = None
        first, end, pre_err = nextPoint(start, drawed, pixel_value)
        print("reset", first, end)
        key = (min(first, end), max(first, end))
        USED_LINES.add(key)
        line = LINE_PIXEL[key]
        drawed = merge_pixel(drawed, line)
    else:
        new_end, err = nextPoint(end, drawed, pixel_value)
        if new_end is None:
            print("find no point")
            break
        key = (min(end, new_end), max(end, new_end))
        print("new end", new_end)
        USED_LINES.add(key)
        line = LINE_PIXEL[key]
        drawed = merge_pixel(drawed, line)
        end = new_end
        pre_err = err
    if i % N == 0:
        frame = drawed.copy()
        frame = 255 * np.ones_like(frame) - frame
        frames.append([plt.imshow(frame, cmap='gray', animated=True)])

plt.imsave("./lines.png", drawed)
ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=False)
plt.show()
