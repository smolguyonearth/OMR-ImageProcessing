import os
import re
import json
import cv2
import numpy as np
import pandas as pd
import imutils

# ── Paths ──────────────────────────────────────────────────────────────────
IMG_REF_PATH  = 'synthetic/ref.png'
PROCESSED_DIR = 'synth100/processed_data'
OUTPUT_CSV    = 'result.csv'
# ──────────────────────────────────────────────────────────────────────────

def sort_key(s):
    p = re.match(r'(\d+)([A-Z])', s)
    return int(p.group(1)), p.group(2)


def process_image(img_sample_path, img_ref):
    """Full OMR pipeline for one sample image. Returns dict {q1: 'A', ...}."""

    img_sample = cv2.imread(img_sample_path)
    if img_sample is None:
        raise FileNotFoundError(f'Cannot read: {img_sample_path}')

    # ── Step 3 — SIFT + matching ──────────────────────────────────────────
    gray_ref    = cv2.cvtColor(img_ref,    cv2.COLOR_BGR2GRAY)
    gray_sample = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_ref,    None)
    kp2, des2 = sift.detectAndCompute(gray_sample, None)

    good_matches = []
    for i, d1 in enumerate(des1):
        distances  = np.linalg.norm(des2 - d1, axis=1)
        idx_sorted = np.argsort(distances)
        if distances[idx_sorted[0]] < 0.75 * distances[idx_sorted[1]]:
            good_matches.append(cv2.DMatch(
                _queryIdx=i, _trainIdx=int(idx_sorted[0]),
                _distance=distances[idx_sorted[0]]
            ))

    # ── Step 4 — Homography & registration ───────────────────────────────
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    h_matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    height, width = img_ref.shape[:2]
    registered = cv2.warpPerspective(img_sample, h_matrix, (width, height))

    # ── Step 5 — Shadow removal ───────────────────────────────────────────
    img_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    se       = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg_img   = cv2.medianBlur(cv2.dilate(img_gray, se), 21)
    diff     = 255 - cv2.absdiff(img_gray, bg_img)
    norm     = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # ── Step 6 — CLAHE ───────────────────────────────────────────────────
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(norm)

    # ── Step 7 — Adaptive threshold ──────────────────────────────────────
    thresh_mask = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    # ── Step 8 — Find answer section ─────────────────────────────────────
    img_clean_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def count_omr_circles(roi_img):
        gray   = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        t      = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 5)
        cs = imutils.grab_contours(
            cv2.findContours(t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        )
        return sum(1 for c in cs
                   if 100 <= cv2.contourArea(c) <= 300
                   and 0.5 <= cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]) <= 1.5)

    roi = None
    for cnt in cnts:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w > 100 and h > 50:
            roi_temp = img_clean_bgr[y:y+h, x:x+w]
            if count_omr_circles(roi_temp) >= 236:
                roi = roi_temp.copy()
                break

    if roi is None:
        raise RuntimeError('Answer section not found.')

    # ── Step 9 — Bubble filtering ─────────────────────────────────────────
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed   = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
    thresh_roi = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 5)

    all_bubble_cnts = imutils.grab_contours(
        cv2.findContours(thresh_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    filtered_cnts = [
        c for c in all_bubble_cnts
        if 100 <= cv2.contourArea(c) <= 300
        and 0.5 <= cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]) <= 1.5
    ]

    # ── Step 10 — Grid mapping ────────────────────────────────────────────
    def _group_medians(values, groups):
        if len(values) == 0:
            return []
        if len(values) < groups:
            return list(np.linspace(float(np.min(values)), float(np.max(values)), groups))
        splits = np.array_split(np.sort(values), groups)
        return [float(np.median(s)) for s in splits]

    def build_grid(cnts, rows=15, cols=16):
        if not cnts:
            return []
        centers, widths, heights = [], [], []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            centers.append((x + w / 2.0, y + h / 2.0, x, y, w, h))
            widths.append(w); heights.append(h)
        col_centers = _group_medians([c[0] for c in centers], cols)
        row_centers = _group_medians([c[1] for c in centers], rows)
        median_w = int(np.median(widths))
        median_h = int(np.median(heights))
        grid = {}
        for cx, cy, x, y, w, h in centers:
            ri   = int(np.argmin([abs(cy - r) for r in row_centers]))
            ci   = int(np.argmin([abs(cx - c) for c in col_centers]))
            dist = abs(cy - row_centers[ri]) + abs(cx - col_centers[ci])
            key  = (ri, ci)
            if key not in grid or dist < grid[key]['dist']:
                grid[key] = {'x': x, 'y': y, 'w': w, 'h': h, 'dist': dist}
        boxes = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) in grid:
                    b = grid[(r, c)]
                    boxes.append({'x': int(b['x']), 'y': int(b['y']),
                                  'w': int(b['w']), 'h': int(b['h'])})
                else:
                    boxes.append({'x': int(col_centers[c] - median_w / 2),
                                  'y': int(row_centers[r] - median_h / 2),
                                  'w': median_w, 'h': median_h})
        return boxes

    grid_boxes = build_grid(filtered_cnts)
    if len(grid_boxes) != 240:
        raise RuntimeError(f'Expected 240 grid boxes, got {len(grid_boxes)}')

    # ── Step 11 — OMR map ─────────────────────────────────────────────────
    omr_map = {}
    option_letters = ['A', 'B', 'C', 'D']
    for row_idx in range(15):
        row_bubbles = grid_boxes[row_idx * 16:(row_idx + 1) * 16]
        for block_idx in range(4):
            q_num   = block_idx * 15 + row_idx + 1
            options = row_bubbles[block_idx * 4:block_idx * 4 + 4]
            for opt_idx, bubble in enumerate(options):
                omr_map[f"{q_num}{option_letters[opt_idx]}"] = {
                    k: bubble[k] for k in ['x', 'y', 'w', 'h']
                }

    # ── Step 12 — Filled bubble detection ────────────────────────────────
    bubble_only_vis = np.ones_like(roi, dtype=roi.dtype) * 255
    for box in omr_map.values():
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bubble_only_vis.shape[1], x+w), min(bubble_only_vis.shape[0], y+h)
        if x1 < x2 and y1 < y2:
            bubble_only_vis[y1:y2, x1:x2] = roi[y1:y2, x1:x2].copy()

    gray_section = cv2.cvtColor(bubble_only_vis, cv2.COLOR_BGR2GRAY)
    _, thresh_otsu = cv2.threshold(gray_section, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    final_cnts = imutils.grab_contours(
        cv2.findContours(thresh_otsu.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    candidate_bubbles = [
        c for c in final_cnts
        if cv2.boundingRect(c)[2] >= 12
        and cv2.boundingRect(c)[3] >= 12
        and 0.5 <= cv2.boundingRect(c)[2] / float(cv2.boundingRect(c)[3]) <= 1.5
    ]

    PIXEL_THRESHOLD = 150
    filled_labels   = []
    for c in candidate_bubbles:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        mask_c = np.zeros(thresh_otsu.shape, dtype='uint8')
        cv2.drawContours(mask_c, [c], -1, 255, -1)
        masked   = cv2.bitwise_and(thresh_otsu, thresh_otsu, mask=mask_c)
        total_px = cv2.countNonZero(masked)
        if total_px >= PIXEL_THRESHOLD:
            for label, box in omr_map.items():
                buf = 5
                if (box['x'] - buf <= cx <= box['x'] + box['w'] + buf and
                        box['y'] - buf <= cy <= box['y'] + box['h'] + buf):
                    filled_labels.append(label)
                    break

    filled_labels.sort(key=sort_key)

    # ── Build per-question answer dict ────────────────────────────────────
    answers = {int(lbl[:-1]): lbl[-1] for lbl in filled_labels}
    return {f'q{q}': answers.get(q, 'BLANK') for q in range(1, 61)}


# ══ Main loop ══════════════════════════════════════════════════════════════

img_ref = cv2.imread(IMG_REF_PATH)
assert img_ref is not None, f'Reference image not found: {IMG_REF_PATH}'

image_files = sorted([
    f for f in os.listdir(PROCESSED_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

records = []
for fname in image_files:
    path = os.path.join(PROCESSED_DIR, fname)
    print(f'Processing {fname} ...', end=' ', flush=True)
    try:
        row = {'filename': fname}
        row.update(process_image(path, img_ref))
        records.append(row)
        print('✅')
    except Exception as e:
        print(f'❌  {e}')
        records.append({'filename': fname, **{f'q{q}': 'ERROR' for q in range(1, 61)}})

cols = ['filename'] + [f'q{i}' for i in range(1, 61)]
df   = pd.DataFrame(records, columns=cols)
df.to_csv(OUTPUT_CSV, index=False)
print(f'\nDone. Results saved → {OUTPUT_CSV}  ({len(df)} rows)')