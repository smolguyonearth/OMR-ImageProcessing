# %%
import os
import re
import json
import tarfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imutils
from imutils import contours

# ==========================================
# Helper Functions
# ==========================================

def auto_register2(img_ref_path, img_warped_path):
    """Aligns the warped image to the reference image using SIFT features (much more robust than ORB)."""
    # 1. Load images in Grayscale for feature detection
    img1 = cv2.imread(img_ref_path)
    img2 = cv2.imread(img_warped_path)

    if img1 is None:
        raise ValueError(f"Could not read reference image: {img_ref_path}")
    if img2 is None:
        raise ValueError(f"Could not read target image: {img_warped_path}")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. Initialize SIFT detector (Scale-Invariant, very robust to blur and perspective)
    sift = cv2.SIFT_create()

    # 3. Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # We use KNN to find the top 2 matches for each descriptor for the ratio test
    bf = cv2.BFMatcher()
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

    # Draw top matches for debugging (optional)
    # debug_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # show_image("SIFT Matches", debug_img)

    if len(good_matches) < 10:
        print("Not enough good matches found!")
        return img2
        
    # 6. Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # 7. Find Homography using RANSAC (removes bad outliers automatically)
    # 5.0 is the RANSAC threshold (pixel distance)
    h_matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # 8. Warp the image
    height, width = img1.shape[:2]
    registered_img = cv2.warpPerspective(img2, h_matrix, (width, height))

    return registered_img

def auto_register(img_ref_path, img_warped_path):
    """Aligns the warped image to the reference image using ORB features."""
    # 1. Load images in Grayscale for feature detection
    img1 = cv2.imread(img_ref_path)
    img2 = cv2.imread(img_warped_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)

    # 3. Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 4. Match features using Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # 5. Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # 6. Find Homography using RANSAC (removes bad matches automatically)
    h_matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # 7. Warp the image
    height, width, channels = img1.shape
    registered_img = cv2.warpPerspective(img2, h_matrix, (width, height))

    return h_matrix, registered_img

def preprocess_omr_gentle(image_path):
    """Removes shadows and enhances bubble contrast."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # --- STEP 1: Gentle Shadow Removal ---
    # Use Dilation to expand white areas to cover text
    # Followed by MedianBlur to smooth it out into a single background sheet
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) 
    bg_img = cv2.dilate(img, structuring_element)
    bg_img = cv2.medianBlur(bg_img, 21) # Smooth lighting

    # Calculate difference (maintains contrast better than division)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    # Normalizing: Adjust lighting to full 0-255 range (true white paper)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # --- STEP 2: Enhance Contrast with CLAHE ---
    # This significantly highlights "pencil-filled bubbles" without adding noise
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(norm_img)

    # --- STEP 3: Adaptive Thresholding ---
    # block_size: Must be odd (larger = wider view)
    # C: Value subtracted from mean (Decrease this if lines disappear)
    thresh = cv2.adaptiveThreshold(
        enhanced_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7 # Setting C to 7 (from 15) retains more details
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

    min_area = 100
    max_area = 300
    valid_circles = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            # Check aspect ratio to ensure it isn't overly distorted
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

    centers = []
    widths = []
    heights = []

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
                cx = col_centers[c]
                cy = row_centers[r]
                x = int(cx - (median_w / 2.0))
                y = int(cy - (median_h / 2.0))
                boxes.append({"x": x, "y": y, "w": median_w, "h": median_h})

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
    # hmatrix, registered_img = auto_register(img_ref_path, img_warped_path)
    registered_img = auto_register2(img_ref_path, img_warped_path)
    cv2.imwrite('warp.png', registered_img)

    flat_img, mask = preprocess_omr_gentle('warp.png')
    cv2.imwrite('clean_background.png', flat_img)

    # 2. Extract Main Section
    img = cv2.imread('clean_background.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # NOTE: Threshold might need adjustment here depending on image lighting
    # edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        contour_img = img.copy()
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
            roi_temp = img[y:y+h, x:x+w]
            circle_count = count_omr_circles(roi_temp)
            max_circles = max(max_circles, circle_count)
            
            if circle_count >= 236: 
                cv2.imwrite('target_section.png', roi_temp)
                roi = roi_temp
                found_target = True
                if debug:
                    print(f"✅ SUCCESS! Perfect Section Extracted! ({circle_count} Circles)")
                    show_image('Perfect Section Extracted', roi)
                break # Stop loop once target is found

    if not found_target:
        print("❌ Could not find any section with exactly 240 circles. Found max: ", max_circles)
        return []

    # 3. Format & Map Bubbles
    roi_gray = cv2.imread('target_section.png', cv2.IMREAD_GRAYSCALE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
    thresh = cv2.adaptiveThreshold(closed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    
    bubble_cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = imutils.grab_contours(bubble_cnts)

    min_area = 100
    max_area = 300 
    filtered_cnts = []
    for c in bubble_cnts:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            if 0.5 <= aspect_ratio <= 1.5:
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
                    x = bubble["x"]
                    y = bubble["y"]
                    w = bubble["w"]
                    h = bubble["h"]
                    label = f"{question_num}{letter}"
                    omr_map[label] = {"x": x, "y": y, "w": w, "h": h}

        # Save template for records
        with open('omr_template_map.json', 'w') as f:
            json.dump(omr_map, f, indent=4)
    else:
        print(f"WARNING: Found {len(filtered_cnts)} contours instead of 240. Map mapping may fail.")
        return []

    # 4. Extract Marked Answers
    img_section = cv2.imread('target_section.png')
    gray_section = cv2.cvtColor(img_section, cv2.COLOR_BGR2GRAY)
    thresh_section = cv2.threshold(gray_section, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    final_cnts = cv2.findContours(thresh_section.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_cnts = imutils.grab_contours(final_cnts)
    
    question_bubbles_final = []
    for c in final_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Using 0.5 to 1.5 aspect ratio catches bubbles filled slightly outside the lines
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
        pixel_threshold = 150
        
        if total_pixels >= pixel_threshold:
            matched_label = None
            for label, box in omr_map.items():
                buffer = 5
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


# %%
processed_data_dir = "synth100/processed_data"
labels_csv_path = "synth100/labels.csv"

# Define paths
# Note: Make sure you have a valid reference image at this path
img_ref_path = "synthetic/ref.png"  

if os.path.exists(processed_data_dir):
    file_names = os.listdir(processed_data_dir)
    # sort 'synthetic_sample_10.png'
    file_names.sort(key=lambda x: int(re.match(r"synthetic_sample_(\d+)\.png", x).group(1)))
else:
    file_names = []
    print(f"Directory not found: {processed_data_dir}")

# 2. Process Dataset and Calculate Accuracy
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

        # Map predictions to dictionary {question_num: answer_letter}
        my_answers_dict = {int(item[:-1]): item[-1] for item in predicted_list}

        # Fetch Ground Truth
        ground_truth_rows = df[df['filename'] == pic]
        if ground_truth_rows.empty:
            print(f"No ground truth found for {pic} in labels.csv")
            continue
            
        ground_truth = ground_truth_rows.iloc[0]
        correct_count = 0

        # Grading Loop
        for i in range(1, total_questions + 1):
            col_name = f'q{i}'
            my_ans = my_answers_dict.get(i, 'BLANK')

            if col_name in ground_truth:
                real_ans = ground_truth[col_name]
                if my_ans == real_ans:
                    correct_count += 1

        accuracy = (correct_count / total_questions) * 100
        total_score_final += accuracy
        valid_files_processed += 1

        print(f"Result: {correct_count} / {total_questions} correct")
        print(f"Accuracy for {pic}: {accuracy:.2f}%\n")

    # 3. Final Overall Score
    if valid_files_processed > 0:
        final_average = total_score_final / valid_files_processed
        print("=" * 40)
        print(f"FINAL OVERALL ACCURACY: {final_average:.2f}% (over {valid_files_processed} files)")
        print("=" * 40)
else:
    print("Required labels.csv or processed images not found.")