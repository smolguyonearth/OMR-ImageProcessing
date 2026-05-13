import cv2
import numpy as np
import random
import csv
import os

# ==========================================
# CONFIGURATION
# ==========================================

SKIP_PROBABILITY = 0.03
CENTER_OFFSET_MIN_PX = -1.5
CENTER_OFFSET_MAX_PX = 1.5
BASE_RADIUS_OFFSET_PX = 0

NUM_STROKES_MIN = 10
NUM_STROKES_MAX = 15
STROKE_THICKNESS = 1

SMEAR_OUTSIDE_PROBABILITY = 0.10
SMEAR_EXTRA_LENGTH_MIN = 1.0
SMEAR_EXTRA_LENGTH_MAX = 2.0

PAPER_NOISE_LEVEL = 0.1
BLUR_KERNEL_SIZE = 3
GRAPHITE_COLOR_MIN = 50
GRAPHITE_COLOR_MAX = 80

NUM_SHEETS_TO_GENERATE = 5

# Directories
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "processed_data"

# ==========================================
# LAYOUT DETECTION
# ==========================================

def detect_answer_boxes(image_path, output_path='boxes_detected.png'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_circles = []
    
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        if 80 < area < 400 and 0.8 <= aspect_ratio <= 1.2:
            if y < 450:
                continue

            in_valid_column = False
            for (min_x, max_x) in [(135, 210), (295, 370), (455, 530), (615, 690)]:
                if min_x <= x <= max_x:
                    in_valid_column = True
                    break
                    
            if not in_valid_column:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.6:
                    valid_circles.append((x, y, w, h, c))

    def is_close(r1, r2, dist=10):
        return abs(r1[0] - r2[0]) < dist and abs(r1[1] - r2[1]) < dist
    
    unique_boxes = []
    for (x, y, w, h, c) in valid_circles:
        found = False
        for (ux, uy, uw, uh, uc) in unique_boxes:
            if is_close((x, y), (ux, uy)):
                found = True
                break
        if not found:
            unique_boxes.append((x, y, w, h, c))

    def get_block_index(bx):
        if bx < 250: return 0
        elif bx < 400: return 1
        elif bx < 550: return 2
        else: return 3
        
    unique_boxes.sort(key=lambda b: (get_block_index(b[0]), b[1] // 15, b[0]))
    
    print(f"Detected {len(unique_boxes)} valid circular boxes.")

    output_img = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    
    for (x, y, w, h, c) in unique_boxes:
        block_idx = get_block_index(x)
        color = colors[block_idx]
        center = (x + w // 2, y + h // 2)
        radius = int((w + h) / 4)
        cv2.circle(output_img, center, radius, color, 2)
        
    cv2.imwrite(output_path, output_img)
    return unique_boxes

# ==========================================
# SYNTHETIC GENERATION
# ==========================================

def fill_pencil(img, center, radius):
    padding = 6
    x_min = max(0, center[0] - radius - padding)
    y_min = max(0, center[1] - radius - padding)
    x_max = min(img.shape[1], center[0] + radius + padding)
    y_max = min(img.shape[0], center[1] + radius + padding)
    
    roi = img[y_min:y_max, x_min:x_max].astype(np.float32)
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
    
    cx = center[0] - x_min + random.uniform(CENTER_OFFSET_MIN_PX, CENTER_OFFSET_MAX_PX)
    cy = center[1] - y_min + random.uniform(CENTER_OFFSET_MIN_PX, CENTER_OFFSET_MAX_PX)
    
    base_radius = radius + BASE_RADIUS_OFFSET_PX
    cx_int, cy_int = int(round(cx)), int(round(cy))
    
    cv2.ellipse(mask, (cx_int, cy_int), (base_radius, base_radius + random.randint(0, 1)), 
                random.randint(0, 360), 0, 360, random.uniform(0.6, 0.9), -1)
    
    main_angle = random.uniform(0, np.pi)
    for _ in range(random.randint(NUM_STROKES_MIN, NUM_STROKES_MAX)):
        angle = main_angle + random.uniform(-0.2, 0.2)
        offset_dist = random.uniform(-base_radius, base_radius)
        offset_x = int(np.sin(main_angle) * offset_dist)
        offset_y = int(-np.cos(main_angle) * offset_dist)
        
        length = base_radius + random.uniform(-1, 1)
        if random.random() < SMEAR_OUTSIDE_PROBABILITY:
            length += random.uniform(SMEAR_EXTRA_LENGTH_MIN, SMEAR_EXTRA_LENGTH_MAX)
            
        x1 = int(cx + offset_x + np.cos(angle) * length)
        y1 = int(cy + offset_y + np.sin(angle) * length)
        x2 = int(cx + offset_x - np.cos(angle) * length)
        y2 = int(cy + offset_y - np.sin(angle) * length)
        
        cv2.line(mask, (x1, y1), (x2, y2), random.uniform(0.7, 1.0), STROKE_THICKNESS)

    noise = np.random.normal(0, PAPER_NOISE_LEVEL, mask.shape).astype(np.float32)
    mask = np.clip(mask + noise, 0, 1)
    mask = cv2.GaussianBlur(mask, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)
    mask = np.clip(mask, 0, 1)
    mask = np.power(mask, 1.2)

    c_val = random.randint(GRAPHITE_COLOR_MIN, GRAPHITE_COLOR_MAX)
    color = np.array([c_val + random.randint(-5, 5) for _ in range(3)], dtype=np.float32)
    color = np.clip(color, 0, 255)

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - (1.0 - color[c] / 255.0) * mask)
        
    img[y_min:y_max, x_min:x_max] = np.clip(roi, 0, 255).astype(np.uint8)

def main():
    # Usually you'd reference a base template image.
    # We will assume a 'ref.png' is located in the working directory that you invoke the script from.
    input_image = '../../ref.png'
    if not os.path.exists(input_image):
        # Fallback to current directory for robust execution
        input_image = 'ref.png'
        if not os.path.exists(input_image):
            print(f"Error: {input_image} not found.")
            return

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    output_boxes_path = os.path.join(PROCESSED_DATA_DIR, '../boxes_detected.png')
    boxes = detect_answer_boxes(input_image, output_boxes_path)
    
    items = [boxes[i:i+4] for i in range(0, len(boxes), 4)]
    choice_labels = ['A', 'B', 'C', 'D']
    all_labels = []
    
    for i in range(1, NUM_SHEETS_TO_GENERATE + 1):
        img_copy = cv2.imread(input_image)
        filename = f'synthetic_sample_{i}.png'
        current_filepath = os.path.join(RAW_DATA_DIR, filename)
        sheet_labels = [filename]
        
        for choices in items:
            if random.random() < SKIP_PROBABILITY: 
                sheet_labels.append("BLANK")
                continue
            
            choice_idx = random.randint(0, 3)
            sheet_labels.append(choice_labels[choice_idx])
            
            x, y, w, h, _ = choices[choice_idx]
            center = (x + w // 2, y + h // 2)
            radius = int(max(w, h) / 2)
            fill_pencil(img_copy, center, radius)
            
        cv2.imwrite(current_filepath, img_copy)
        all_labels.append(sheet_labels)
        print(f"Saved synthetic sheet {i} to {current_filepath}")
            
    csv_path = 'labels.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename'] + [f'q{i+1}' for i in range(60)])
        writer.writerows(all_labels)
    
    print(f"Saved labels to {csv_path}")

if __name__ == "__main__":
    main()
