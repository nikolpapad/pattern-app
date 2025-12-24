''' Repository: https://github.com/nikolpapad/grid_extraction.git
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from utils import extend_line, classify_cell, refine_line_positions, color_all_cells, rle_labels, rle_to_instructions
from utils import generate_instructions

img_path =r"C:\Users\nikol\Downloads\page_5.png"
img = cv2.imread(img_path)

# Gray scale + light blur to keep edges 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur1 = cv2.GaussianBlur(gray, (5,5), 0)
gray_blur = cv2.GaussianBlur(gray_blur1, (5,5), 0)

otsu = cv2.threshold(
    gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]  # Binarize with Otsu

erosion = cv2.erode(otsu, np.ones((5, 5), np.uint8), iterations=1)
dilation = cv2.dilate(erosion, np.ones((3, 3), np.uint8), iterations=1)
binary_copy = cv2.bitwise_not(dilation)  # Invert: grid lines are white

orig = img.copy() # original grayscale or color image
height, width = binary_copy.shape

edges = cv2.Canny(binary_copy, 100, 150, apertureSize=3)

# Standard Hough Line 
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=20, maxLineGap=40)
if lines is None:
    raise RuntimeError("No lines detected. Tune parameters or preprocessing.")
else:
    print(f"\nDetected {0 if lines is None else len(lines)} lines")

# ----------- Separate lines into horizontal and vertical + cluster -------------
atol = 0.1  # angle tolerance in degrees
pixel_tol = 5

hor_lines = {}
vert_lines = {}

from collections import defaultdict
horizontal_groups = defaultdict(list)
vertical_groups = defaultdict(list)

for x1, y1, x2, y2 in tqdm(lines[:, 0, :], desc="Processing detected lines:"):
    # Check if angle is straight enough
    dx, dy = (x2 - x1), (y2 - y1)
    angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
    near_horizontal = min(angle, 180.0 - angle) < atol
    near_vertical   = abs(angle - 90.0) < atol

    if not (near_horizontal or near_vertical):
        continue

    x_start, y_start, x_end, y_end = extend_line(height, width, x1, y1, x2, y2)

    if near_horizontal:
        key = int(round(y_start / pixel_tol))
        if key not in hor_lines:
            hor_lines[key] = (x_start, y_start, x_end, y_end)
            horizontal_groups[key].append(y_start)
    elif near_vertical:
        key = int(round(x_start / pixel_tol))
        if key not in vert_lines:
            vert_lines[key] = (x_start, y_start, x_end, y_end)
            vertical_groups[key].append(x_start)

    # for (x_start, y_start, x_end, y_end) in list(hor_lines.values()) + list(vert_lines.values()):
    #     cv2.line(orig, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)

# ## Original image + detected lines overlay
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
# plt.title("Detected Grid Lines Overlay")

###################################################################################################
  
# unique x for each vertical line, unique y for each horizontal line
xs_raw = [int(np.median(group)) for group in vertical_groups.values()]
ys_raw = [int(np.median(group)) for group in horizontal_groups.values()]
if not xs_raw or not ys_raw:
    raise RuntimeError("No grid lines found:Check detection / thresholds.")

# Refine by merging nearby detections (thickness vs gap)
xs_refined = refine_line_positions(xs_raw)
ys_refined = refine_line_positions(ys_raw)

xs_raw = xs_refined
ys_raw = ys_refined

grid_left,grid_right = xs_raw[0], xs_raw[-1]
grid_top, grid_bottom = ys_raw[0], ys_raw[-1]

grid_width  = grid_right - grid_left
grid_height = grid_bottom - grid_top
n_cols, n_rows = len(xs_raw) - 1, len(ys_raw) - 1 # for 3 vertical lines -> 2 columns

reconstructed = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
debug_cells = color_all_cells(reconstructed, xs_raw, ys_raw, n_cols, n_rows, grid_left, grid_top,orig, plotting = False)


cell_margin = 5 # margin to avoid sampling the black grid lines

# Pattern colors per cell!!!
pattern_colors, pattern_labels = [], []
for j in tqdm(range(n_rows), desc="Paint that cell area in the reconstructed image...."):
    row_colors, row_clabels = [], []
    y1 = ys_raw[j]
    y2 = ys_raw[j + 1]

    for i in range(n_cols):
        x1 = xs_raw[i]
        x2 = xs_raw[i + 1]

        # ----- Crop original cell (inside the grid) -----
        yy1 = max(y1 + cell_margin, 0)
        yy2 = min(y2 - cell_margin, height)
        xx1 = max(x1 + cell_margin, 0)
        xx2 = min(x2 - cell_margin, width)

        cell = img[yy1:yy2, xx1:xx2]

        if cell.size == 0:
            color = np.array([5, 180, 85], dtype=np.uint8)
            color_label = "white"
        else:
            mean_color = cell.reshape(-1, 3).mean(axis=0).astype(np.float32)
            color, clabel = classify_cell(mean_color)
            if clabel == "other": 
                print(f"Cell ({i},{j}) mean color: {mean_color}, classified as {clabel}")
                
        row_colors.append(color)
        row_clabels.append(clabel)
        # ----- Paint that cell area in the reconstructed image -----
        new_y1 = y1 - grid_top
        new_y2 = y2 - grid_top
        new_x1 = x1 - grid_left
        new_x2 = x2 - grid_left

        reconstructed[new_y1:new_y2, new_x1:new_x2] = color

    pattern_colors.append(row_colors) # can be skipped
    pattern_labels.append((rle_labels(row_clabels)))

print(f"--------\nGrid size: width={grid_width}, height={grid_height}. \nGrid cells: {n_rows} rows x {n_cols} cols \n--------")
instructions = generate_instructions(pattern_labels, start_corner="bottom-left", crochet=False)
print("-------------------------------------------------------------------\nPattern Instructions:")
for instr in instructions:
    print(instr)

plt.figure(figsize=(4, 4))
plt.subplot(1, 3, 1),plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)),plt.title("Original grid area"),plt.axis("off")
plt.subplot(1, 3, 2),plt.imshow(cv2.cvtColor(debug_cells, cv2.COLOR_BGR2RGB)),plt.title("Each detected cell = different color"),plt.axis("off")
plt.subplot(1, 3, 3),plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)),plt.title("Reconstructed pattern"),plt.axis("off")
plt.show()


