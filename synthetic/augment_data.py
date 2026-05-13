import cv2
import numpy as np
import albumentations as A
import os
import glob
from augraphy import AugraphyPipeline, LightingGradient, ShadowCast, NoiseTexturize

# ==========================================
# CONFIGURATION (DIALED BACK)
# ==========================================

INPUT_DIR = "raw_data"
OUTPUT_DIR = "processed_data"

# 0. Background Padding (Simulates white desk)
MIN_BACKGROUND_MARGIN_PERCENT = 0.15 # Adds 10% extra space around the paper
MAX_BACKGROUND_MARGIN_PERCENT = 0.25 # Adds 15% extra space around the paper

# 1. Micro-Warping (Very subtle paper curves)
GRID_DISTORTION_PROB = 1.0
GRID_DISTORTION_LIMIT = 0.03

# 2. Perspective (Camera tilt)
PERSPECTIVE_PROB = 1.0
PERSPECTIVE_SCALE = 0.04 # Reduced severity so it doesn't skew too hard

# 3. Lens Physics (Smartphone bulge & depth of field)
OPTICAL_DISTORTION_PROB = 1.0
BARREL_DISTORTION_LIMIT = 0.05 # Halved 
DOF_BLUR_KERNEL_SIZE = 5       # Drastically reduced (was 15)
DOF_BLUR_INTENSITY = 0.3       # Max blur is only 30% opaque at the top edge

# 4. Environment & Noise (Augraphy)
LIGHTING_GRADIENT_PROB = 0.8
SHADOW_CAST_PROB = 0.6
SENSOR_NOISE_PROB = 0.00 # Reduced frequency of noise

# ==========================================
# CUSTOM FUNCTIONS
# ==========================================

def add_white_background(image, min_margin_percent, max_margin_percent):
    """
    Places the paper on a larger white canvas before warping.
    This prevents the edges from being cropped out during perspective shifts.
    """
    h, w = image.shape[:2]
    margin_y = int(h * np.random.uniform(min_margin_percent, max_margin_percent))
    margin_x = int(w * np.random.uniform(min_margin_percent, max_margin_percent))
    r= 205 + np.random.randint(-10, 10)
    g= 205 + np.random.randint(-10, 10)
    b= 200 + np.random.randint(-10, 10)
    
    # Pad with random range of off-white to add slight variation in background color
    padded_image = cv2.copyMakeBorder(
        image, 
        top=margin_y, bottom=margin_y, left=margin_x, right=margin_x, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[r, g, b]
    )
    return padded_image

def apply_depth_of_field_blur(image, max_ksize=5, max_intensity=0.3):
    """
    A much softer depth of field blur. 
    """
    if max_ksize % 2 == 0: max_ksize += 1
        
    blurred = cv2.GaussianBlur(image, (max_ksize, max_ksize), 0)
    h, w = image.shape[:2]
    
    # Gradient only goes up to max_intensity (e.g., 0.3) instead of 1.0
    gradient = np.linspace(max_intensity, 0.0, h).reshape(h, 1, 1)
    
    dof_image = (blurred * gradient + image * (1.0 - gradient)).astype(np.uint8)
    return dof_image

# ==========================================
# PIPELINE DEFINITIONS
# ==========================================

geometry_pipeline = A.Compose([
    A.GridDistortion(
        num_steps=5, 
        distort_limit=GRID_DISTORTION_LIMIT, 
        p=GRID_DISTORTION_PROB
    ),
    A.Perspective(
        scale=(0.02, PERSPECTIVE_SCALE), 
        keep_size=True, 
        pad_mode=cv2.BORDER_CONSTANT, 
        pad_val=(255, 255, 255), # Ensure perspective padding is also white
        p=PERSPECTIVE_PROB
    ),
    A.OpticalDistortion(
        distort_limit=BARREL_DISTORTION_LIMIT, 
        shift_limit=0.05, 
        p=OPTICAL_DISTORTION_PROB
    )
])

# Much softer physics pipeline
physics_pipeline = AugraphyPipeline(
    ink_phase=[],
    paper_phase=[],
    post_phase=[
        ShadowCast(p=SHADOW_CAST_PROB),
        LightingGradient(
            light_position=None, 
            direction=None, 
            max_brightness=255, 
            min_brightness=150, # Raised minimum brightness so it doesn't get too dark
            mode="linear", 
            p=LIGHTING_GRADIENT_PROB
        ),
    ]
)

# ==========================================
# EXECUTION
# ==========================================

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not find image {image_path}.")
        return

    # 1. Place on white background FIRST
    image_on_bg = add_white_background(image, MIN_BACKGROUND_MARGIN_PERCENT, MAX_BACKGROUND_MARGIN_PERCENT)

    # Convert to RGB for Albumentations
    image_rgb = cv2.cvtColor(image_on_bg, cv2.COLOR_BGR2RGB)

    # 2. Geometry & Optics
    geom_result = geometry_pipeline(image=image_rgb)["image"]
    
    # 3. Soft Depth of Field
    dof_result = apply_depth_of_field_blur(
        geom_result, 
        max_ksize=DOF_BLUR_KERNEL_SIZE, 
        max_intensity=DOF_BLUR_INTENSITY
    )

    # 4. Soft Lighting & Shadows
    image_bgr_for_aug = cv2.cvtColor(dof_result, cv2.COLOR_RGB2BGR)
    final_result = physics_pipeline(image_bgr_for_aug)

    cv2.imwrite(output_path, final_result)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    
    if not images:
        print(f"No images found in {INPUT_DIR}.")
        return
        
    for image_path in images:
        filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_DIR, filename)
        print(f"Processing {filename}...")
        process_image(image_path, output_path)
        
    print(f"Done! Processed {len(images)} images to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
