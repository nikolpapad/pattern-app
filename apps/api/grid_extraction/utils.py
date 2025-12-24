import fitz  # PyMuPDF
import os
from PIL import Image
import io
import math
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2

#  Take pdf five back images
def extractFromPDF(pdf_path, out_dir = None):
    if out_dir is None:
        out_dir = os.path.splitext(pdf_path)[0] + "_pages"
    os.makedirs(out_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for i, page in tqdm(enumerate(doc, start=1), total = total_pages , desc="Extracting pages into images:"):
            out = os.path.join(out_dir, f"page_{i}.png")
            
            if os.path.exists(out):
                tqdm.write(f"Skipped existing: {out}")
                continue

            pix = page.get_pixmap(dpi=400)     # high quality
            img_bytes = pix.tobytes("png")

            img = Image.open(io.BytesIO(img_bytes))
            img.save(out, format='PNG')
            

# # pdf_path = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\C-TT006.pdf"
# # out_dir = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\deutero_imgs"
# extractFromPDF(pdf_path, out_dir)

def extend_line(height, width, x1, y1, x2, y2, SCALE=10):
    """
    Extend the segment (x1,y1)-(x2,y2) in both directions and clip to image.
    Always returns 4 ints: (x_start, y_start, x_end, y_end).
    height, width: image dimensions.
    """
    # Image dims
    h, w = height, width
    distance = SCALE * max(w, h)

    # Work with Python ints
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        # degenerate line; just clamp a single point
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        return x1, y1, x1, y1

    # Normalize direction
    ux = dx / length
    uy = dy / length

    # Extend in both directions
    p3_x = int(round(x1 - ux * distance))
    p3_y = int(round(y1 - uy * distance))
    p4_x = int(round(x2 + ux * distance))
    p4_y = int(round(y2 + uy * distance))

    # Clip to image boundaries
    def clip_point(x, y):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return x, y

    p3_x, p3_y = clip_point(p3_x, p3_y)
    p4_x, p4_y = clip_point(p4_x, p4_y)

    return p3_x, p3_y, p4_x, p4_y

def classify_cell(mean_color):
    """
    Classify the mean color into basic color categories.
    mean_color: tuple of (B, G, R) values.
    Returns a string representing the color category.
    Threshols:
    - <50 -> black
    - >200 and low spread -> white
    - high R, low G,B -> red
    """
    b, g, r = map(float, mean_color)
    gray_weighted_average= 0.299 * r + 0.587 * g + 0.114 * b
    spread = max(r,g,b) - min(r,g,b)

    th_black = 20
    th_white = 150
    th_gray_spread = 15 # max spread for grayish colors
    th_red = 40 # min difference R - max(G,B) to be red
    th_dark = 100 

    if gray_weighted_average < th_black: # Very dark doesn't matter the spread
        return (np.array([0,0,0], dtype=np.uint8)), "black"
    elif gray_weighted_average > th_white and spread < th_gray_spread: # very bright and low spread
        return (np.array([255,255,255], dtype=np.uint8)), "white"
    elif (r - max(g,b)) > th_red and gray_weighted_average < th_dark:
        return (np.array([0,0,153], dtype=np.uint8)), "dark_red"
    else:
        return (np.array([255,255,255], dtype=np.uint8)), "white"  # light gray treated as white
    
def refine_line_positions(raw_positions):
    """
    Take a list of raw line coordinates (e.g. xs_raw or ys_raw),
    and merge positions that are closer than the estimated line thickness.
    Returns a cleaned, sorted list of unique line positions.
    """
    if len(raw_positions) <= 1:
        return sorted(raw_positions)

    positions = np.array(sorted(raw_positions), dtype=np.float32)
    diffs = np.diff(positions)

    # If all diffs are zero-ish, just return the median
    if np.all(diffs == 0):
        return [int(np.median(positions))]

    # Sort diffs to separate "small" (thickness) vs "big" (cell gaps)
    sorted_diffs = np.sort(diffs)
    k = max(1, int(0.3 * len(sorted_diffs)))  # first 30% as "small"

    small_diffs = sorted_diffs[:k]
    if len(small_diffs) == 0:
        thickness_est = 0
    else:
        thickness_est = float(np.median(small_diffs))

    # Tolerance for grouping positions that belong to the same physical line
    # Some safety factor around the estimated thickness
    max_group_dist = max(2.0, 1.5 * thickness_est)

    merged = []
    current_group = [positions[0]]

    for p in positions[1:]:
        if abs(p - current_group[-1]) <= max_group_dist:
            current_group.append(p)
        else:
            merged.append(int(np.median(current_group)))
            current_group = [p]

    merged.append(int(np.median(current_group)))

    return sorted(merged)
# ------------------------------------------------------------------------------

def color_all_cells(reconstructed, xs_raw, ys_raw, n_cols, n_rows, grid_left, grid_top, orig ,plotting = False):
    debug_cells = np.zeros_like(reconstructed, dtype=np.uint8)

    rng = np.random.default_rng(0)  # fixed seed for reproducibility

    for j in range(n_rows):
        y1 = ys_raw[j]
        y2 = ys_raw[j + 1]

        for i in range(n_cols):
            x1 = xs_raw[i]
            x2 = xs_raw[i + 1]

            # random BGR color for this cell
            rand_color = rng.integers(0, 256, size=3, dtype=np.uint8)

            # map to grid-local coordinates
            new_y1 = y1 - grid_top
            new_y2 = y2 - grid_top
            new_x1 = x1 - grid_left
            new_x2 = x2 - grid_left

            debug_cells[new_y1:new_y2, new_x1:new_x2] = rand_color
    return debug_cells

def rle_labels(labels):

    """
    Run-length encode a list of labels.
    Returns a list of (label, count) tuples.
    """
    if not labels:
        return []
    runs = []
    current_label = labels[0]
    count = 1

    for lbl in labels[1:]:
        if lbl == current_label:
            count += 1
        else:
            runs.append((current_label, count))
            current_label = lbl
            count = 1 # reset 

    runs.append((current_label, count))
    return runs



def rle_to_instructions(runs):
    """
    Docstring for rle_to_instructions
    
    :param runs: list of (label, count) tuples, ex. [("white", 5), ("dark_red", 1)]
    :return: list of count and color strings, ex. 5 White, 1 Red
    """
    pretty = {
            "white": "White",
            "black": "Black",
            "dark_red": "Red",
        }
    
    parts = []

    for labels, count in runs:      
        color = pretty.get(labels, labels)
        if count == 1:
            parts.append(str(f"1 {color}"))
        else:
            parts.append(str(f"{count} {color}s"))
    return ", ".join(parts)

def generate_instructions(pattern_labels, start_corner="bottom-left", crochet=False):

    instructions = []
    n = len(pattern_labels) 
    if n == 0:
        return []
    # 1) Determine vertical order: bottom to top? #
    start_corner = start_corner.lower()
    if start_corner.startswith("bottom"):
        row_indices = (n-1, -1, -1)  # bottom to top
    elif start_corner.startswith("top"):
        row_indices = (0, n, 1)  # top to bottom
    else:
        raise ValueError("start_corner must be 'bottom-left' or 'top-left'")
    # 2) Generate instructions for horizontal order: right or left? #
    if start_corner.endswith("left"):
        base_direction = "left-to-right"
    elif start_corner.endswith("right"):
        base_direction = "right-to-left"
    else:
        raise ValueError("start_corner must be 'bottom-left' or 'top-left'")
    # 3) Generate instructions #
    for row_from_bottom, row_index in enumerate(range(row_indices[0], row_indices[1], row_indices[2]), start=1):
        row = pattern_labels[row_index]

        if not crochet:
            hor_direction = base_direction
        else:
            if row_from_bottom % 2 == 1: 
            # if row_index % 2 == 1:
                hor_direction = base_direction
            else:
                if base_direction == "left-to-right":
                    hor_direction = "right-to-left"
                else:
                    hor_direction = "left-to-right"
        
        
        if hor_direction == "left-to-right":
            labels = row  # left to right
        else:
            labels = row[::-1] # Right to left

        runs = rle_labels(labels)
        instr = rle_to_instructions(runs)
        instructions.append(
            f"Row {row_from_bottom} (start {start_corner}): {instr}"
            )
    return instructions
        

    


