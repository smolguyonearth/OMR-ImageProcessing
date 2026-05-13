# Synthetic Data Generator

This directory contains standalone scripts for generating synthetic OMR (Optical Mark Recognition) sheets. 

## Structure
- `generate_data.py`: A standalone script that integrates layout detection of bubble options and synthetic generation of pencil strokes. 
- `augment_data.py`: A script that applies realistic physical document augmentation (e.g. lighting, shadow, camera perspective, blur) to the generated images in `raw_data`.
- `labels.csv`: The label file generated alongside the sheets.
- `raw_data/`: Contains the generated unmodified synthetic OMR images.
- `processed_data/`: Stores the processed versions of the data augmented by `augment_data.py`.

## Usage
1. Provide a base template `ref.png` in the directory you are invoking the script from.
2. Run `python generate_data.py` to create the initial sheets. This outputs images to `raw_data` and writes `labels.csv`.
3. Run `python augment_data.py` to apply physical augmentation to the images in `raw_data`, outputting them to `processed_data`.
4. Adjust the global configuration inside the scripts to tune effects.
