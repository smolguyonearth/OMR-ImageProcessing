import os
import re
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils

# ==========================================
# Helper Functions
# ==========================================

def auto_register(img_ref_path, img_warped_path):
    """Aligns the warped image to the reference image using SIFT features."""
    img1 = cv2.imread(img_ref_path)
    img2 = cv2.imread(img_warped_path)

    if img1 is None:
        raise ValueError(f"Could not read reference image: {img_ref_path}")
    if img2 is None:
        raise ValueError(f"Could not read target image: {img_warped_path}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 4. Match 
    good_matches = []
    
    for i, d1 in enumerate(des1):
        distances = np.linalg.norm(des2 - d1, axis=1)
        
        idx_sorted = np.argsort(distances)
        best_idx = idx_sorted[0] 
        second_best_idx = idx_sorted[1]
        
        dist_m = distances[best_idx]
        dist_n = distances[second_best_idx]
        
        if dist_m < 0.75 * dist_n:
            match_obj = cv2.DMatch(_queryIdx=i, _trainIdx=int(best_idx), _distance=dist_m)
            good_matches.append(match_obj)

    if len(good_matches) < 10:
        print("Not enough good matches found!")
        return img2
        
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find Homography using RANSAC
    h_matrix, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Warp the image
    height, width = img1.shape[:2]
    registered_img = cv2.warpPerspective(img2, h_matrix, (width, height))

    return registered_img

def preprocess_omr_gentle(img):
    """Removes shadows and enhances bubble contrast. Accepts an image array."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- STEP 1: Gentle Shadow Removal ---
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
    bg_img = cv2.dilate(gray, structuring_element)
    bg_img = cv2.medianBlur(bg_img, 21)

    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # --- STEP 2: Enhance Contrast with CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(norm_img)

    # --- STEP 3: Adaptive Thresholding ---
    thresh = cv2.adaptiveThreshold(
        enhanced_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    return enhanced_img, thresh

def count_omr_circles(roi_img):
    """Counts valid OMR circles within a Region of Interest."""
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    valid_circles = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 100 <= area <= 300:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 0.5 <= aspect_ratio <= 1.5:
                valid_circles += 1 

    return valid_circles

def _group_medians(values, groups):
    if len(values) == 0:
        return []
    if len(values) < groups:
        return list(np.linspace(float(np.min(values)), float(np.max(values)), groups))
    sorted_vals = np.sort(values)
    splits = np.array_split(sorted_vals, groups)
    return [float(np.median(s)) for s in splits]

def build_grid_from_contours(cnts, rows=15, cols=16):
    if len(cnts) == 0:
        return []

    centers, widths, heights = [], [], []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        centers.append((cx, cy, x, y, w, h))
        widths.append(w)
        heights.append(h)

    col_centers = _group_medians([c[0] for c in centers], cols)
    row_centers = _group_medians([c[1] for c in centers], rows)

    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    grid = {}
    for cx, cy, x, y, w, h in centers:
        row_idx = int(np.argmin([abs(cy - r) for r in row_centers]))
        col_idx = int(np.argmin([abs(cx - c) for c in col_centers]))
        dist = abs(cy - row_centers[row_idx]) + abs(cx - col_centers[col_idx])
        key = (row_idx, col_idx)
        if key not in grid or dist < grid[key]["dist"]:
            grid[key] = {"x": x, "y": y, "w": w, "h": h, "dist": dist}

    boxes = []
    for r in range(rows):
        for c in range(cols):
            key = (r, c)
            if key in grid:
                box = grid[key]
                boxes.append({"x": int(box["x"]), "y": int(box["y"]), "w": int(box["w"]), "h": int(box["h"])})
            else:
                cx_c = col_centers[c]
                cy_c = row_centers[r]
                boxes.append({
                    "x": int(cx_c - (median_w / 2.0)), 
                    "y": int(cy_c - (median_h / 2.0)), 
                    "w": median_w, 
                    "h": median_h
                })

    return boxes

def show_image(title, img, cmap=None):
    """Helper function to plot images."""
    plt.figure(figsize=(10, 8))
    if cmap is None and len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def sort_key(s):
    """Sorts alphanumeric labels (e.g., '1A', '2B')."""
    parts = re.match(r"(\d+)([A-Z])", s)
    return int(parts.group(1)), parts.group(2)

# ==========================================
# Main OMR Pipeline
# ==========================================

def omr_pipeline(img_ref_path, img_warped_path, debug=False):
    """Processes a single OMR image and extracts marked answers."""
    
    # 1. Preprocessing & Registration
    registered_img = auto_register(img_ref_path, img_warped_path)
    flat_img, mask = preprocess_omr_gentle(registered_img)

    # 2. Extract Main Section
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        contour_img = registered_img.copy()
        cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
        show_image('Contours Detected', contour_img)

    found_target = False
    roi = None
    max_circles = 0

    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        
        if w > 100 and h > 50:
            roi_temp = flat_img[y:y+h, x:x+w]
            circle_count = count_omr_circles(roi_temp)
            max_circles = max(max_circles, circle_count)
            
            if circle_count >= 236: 
                roi = roi_temp
                found_target = True
                if debug:
                    print(f"✅ SUCCESS! Perfect Section Extracted! ({circle_count} Circles)")
                    show_image('Perfect Section Extracted', roi)
                break 

    if not found_target:
        print(f"❌ Could not find any section with exactly 240 circles. Found max: {max_circles}")
        return []

    # 3. Format & Map Bubbles
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
    thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    
    bubble_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = imutils.grab_contours(bubble_cnts)

    filtered_cnts = []
    for c in bubble_cnts:
        area = cv2.contourArea(c)
        if 100 <= area <= 300:
            (x, y, w, h) = cv2.boundingRect(c)
            if 0.5 <= (w / float(h)) <= 1.5:
                filtered_cnts.append(c)

    omr_map = {}
    grid_boxes = []
    if len(filtered_cnts) >= 236:
        grid_boxes = build_grid_from_contours(filtered_cnts, rows=15, cols=16)

    if len(grid_boxes) == 240:
        option_letters = ['A', 'B', 'C', 'D']
        for row_idx in range(15):
            row_bubbles = grid_boxes[row_idx * 16:(row_idx + 1) * 16]
            for block_idx in range(4):
                question_num = (block_idx * 15) + row_idx + 1
                start_bubble = block_idx * 4
                options = row_bubbles[start_bubble:start_bubble + 4]

                for opt_idx, bubble in enumerate(options):
                    letter = option_letters[opt_idx]
                    label = f"{question_num}{letter}"
                    omr_map[label] = {
                        "x": bubble["x"], "y": bubble["y"], 
                        "w": bubble["w"], "h": bubble["h"]
                    }

        # Save template for records
        with open('omr_template_map.json', 'w') as f:
            json.dump(omr_map, f, indent=4)
    else:
        print(f"WARNING: Found {len(filtered_cnts)} contours instead of 240. Map mapping may fail.")
        return []

    # 4. Extract Marked Answers
    img_section = roi.copy()
    if len(img_section.shape) == 3:
        gray_section = cv2.cvtColor(img_section, cv2.COLOR_BGR2GRAY)
    else:
        gray_section = img_section.copy()
    _, thresh_section = cv2.threshold(gray_section, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    final_cnts = cv2.findContours(thresh_section.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_cnts = imutils.grab_contours(final_cnts)
    
    question_bubbles_final = []
    for c in final_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 12 and h >= 12 and 0.5 <= ar <= 1.5: 
            question_bubbles_final.append(c)

    filled_labels = []
    for c in question_bubbles_final:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + (w // 2)
        cy = y + (h // 2)
        
        mask = np.zeros(thresh_section.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.bitwise_and(thresh_section, thresh_section, mask=mask)
        
        total_pixels = cv2.countNonZero(mask)
        
        if total_pixels >= 150:
            matched_label = None
            buffer = 5
            for label, box in omr_map.items():
                if (box['x'] - buffer <= cx <= box['x'] + box['w'] + buffer) and \
                   (box['y'] - buffer <= cy <= box['y'] + box['h'] + buffer):
                    matched_label = label
                    break
            
            if matched_label:
                filled_labels.append(matched_label)
                cv2.rectangle(img_section, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_section, matched_label, (x - 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    filled_labels.sort(key=sort_key)
    
    if debug:
        print(f"Successfully identified {len(filled_labels)} labeled answers!")
        show_image("Labeled Student Answers", img_section)

    return filled_labels

# ==========================================
# Execution Script
# ==========================================
def main():
    processed_data_dir = "synth100/processed_data"
    labels_csv_path = "synth100/labels.csv"
    img_ref_path = "synthetic/ref.png"  

    if os.path.exists(processed_data_dir):
        file_names = os.listdir(processed_data_dir)
        file_names.sort(key=lambda x: int(re.match(r"synthetic_sample_(\d+)\.png", x).group(1)))
    else:
        file_names = []
        print(f"Directory not found: {processed_data_dir}")
        return

    if os.path.exists(labels_csv_path) and file_names:
        df = pd.read_csv(labels_csv_path)
        total_score_final = 0
        total_questions = 60
        valid_files_processed = 0

        for pic in file_names:
            path = os.path.join(processed_data_dir, pic)
            print("=" * 40)
            print(f"Processing: {pic}")
            print("=" * 40)

            predicted_list = omr_pipeline(img_ref_path, path, debug=False)

            if not predicted_list:
                print(f"Skipping {pic} due to processing failure.")
                continue

            my_answers_dict = {int(item[:-1]): item[-1] for item in predicted_list}

            ground_truth_rows = df[df['filename'] == pic]
            if ground_truth_rows.empty:
                print(f"No ground truth found for {pic} in labels.csv")
                continue
                
            ground_truth = ground_truth_rows.iloc[0]
            correct_count = 0

            for i in range(1, total_questions + 1):
                col_name = f'q{i}'
                my_ans = my_answers_dict.get(i, 'BLANK')

                if col_name in ground_truth:
                    if my_ans == ground_truth[col_name]:
                        correct_count += 1

            accuracy = (correct_count / total_questions) * 100
            total_score_final += accuracy
            valid_files_processed += 1

            print(f"Result: {correct_count} / {total_questions} correct")
            print(f"Accuracy for {pic}: {accuracy:.2f}%\n")

        if valid_files_processed > 0:
            final_average = total_score_final / valid_files_processed
            print("=" * 40)
            print(f"FINAL OVERALL ACCURACY: {final_average:.2f}% (over {valid_files_processed} files)")
            print("=" * 40)
    else:
        print("Required labels.csv or processed images not found.")

if __name__ == "__main__":
    main()
