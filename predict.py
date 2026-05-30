"""Prediction Pipeline for Test Images (251-347)
Run this when test images and masks are available.
Usage: python predict.py [test_dir] [model_path] [output_path]
"""

"""Feature Extraction for ML Project: To Bee or Not To Bee (Optimized)"""
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
from skimage.measure import moments_hu
import warnings
import pickle
import sys
import glob
warnings.filterwarnings('ignore')

BASE_DIR = r'C:\Users\13968\Documents\ML机器学习\train'
MASKS_DIR = os.path.join(BASE_DIR, 'masks')
SCALE = 0.2  # Downsample to 20% (1200x800 from 6000x4000)

def load_image_and_mask(img_id):
    img_path = os.path.join(BASE_DIR, f'{img_id}.JPG')
    mask_path = os.path.join(MASKS_DIR, f'binary_{img_id}.tif')
    
    img_pil = Image.open(img_path)
    w, h = img_pil.size
    new_size = (int(w * SCALE), int(h * SCALE))
    img_pil = img_pil.resize(new_size, Image.LANCZOS)
    img_rgb = np.array(img_pil)
    
    mask = None
    if os.path.exists(mask_path):
        mask_pil = Image.open(mask_path)
        mask_pil = mask_pil.resize(new_size, Image.NEAREST)
        mask_full = np.array(mask_pil)
        if mask_full.ndim == 3:
            mask = (mask_full[:,:,0] > 0).astype(np.uint8)
        else:
            mask = (mask_full > 0).astype(np.uint8)
    return img_rgb, mask

def compute_symmetry_features(mask):
    if mask is None or mask.sum() == 0:
        return {'sym_h': 0, 'sym_v': 0, 'sym_aspect': 0}
    h, w = mask.shape
    left = mask[:, :w//2]
    right_flipped = np.fliplr(mask[:, w//2:])
    min_w = min(left.shape[1], right_flipped.shape[1])
    left, right_flipped = left[:, :min_w], right_flipped[:, :min_w]
    inter_h = np.logical_and(left, right_flipped).sum()
    union_h = np.logical_or(left, right_flipped).sum()
    sym_h = inter_h / union_h if union_h > 0 else 0
    top = mask[:h//2, :]
    bottom_flipped = np.flipud(mask[h//2:, :])
    min_h = min(top.shape[0], bottom_flipped.shape[0])
    top, bottom_flipped = top[:min_h, :], bottom_flipped[:min_h, :]
    inter_v = np.logical_and(top, bottom_flipped).sum()
    union_v = np.logical_or(top, bottom_flipped).sum()
    sym_v = inter_v / union_v if union_v > 0 else 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.sum() and cols.sum():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        sym_aspect = (ymax - ymin + 1) / (xmax - xmin + 1)
    else:
        sym_aspect = 0
    return {'sym_h': round(sym_h, 6), 'sym_v': round(sym_v, 6), 'sym_aspect': round(sym_aspect, 6)}

def compute_shape_features(mask):
    zero_result = {f'hu{i}': 0 for i in range(1, 8)}
    zero_result.update({'solidity': 0, 'eccentricity': 0, 'extent': 0, 'perimeter': 0, 'area': 0, 'convex_area': 0, 'compactness': 0})
    if mask is None or mask.sum() == 0:
        return zero_result
    mask_u8 = mask.astype(np.uint8)
    hu = moments_hu(mask_u8.astype(float))
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return zero_result
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    convex_area = cv2.contourArea(hull)
    solidity = area / convex_area if convex_area > 0 else 0
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if len(c) >= 5:
        (cx, cy), (major, minor), _ = cv2.fitEllipse(c)
        eccentricity = np.sqrt(1 - (min(major, minor) / max(major, minor)) ** 2) if max(major, minor) > 0 else 0
    else:
        eccentricity = 0
    x, y, bw, bh = cv2.boundingRect(c)
    extent = area / (bw * bh) if bw * bh > 0 else 0
    result = {f'hu{i}': round(hu_log[i - 1], 6) for i in range(1, 8)}
    result.update({'solidity': round(solidity, 6), 'eccentricity': round(eccentricity, 6), 'extent': round(extent, 6), 'perimeter': round(perimeter, 2), 'area': round(area, 2), 'convex_area': round(convex_area, 2), 'compactness': round(compactness, 6)})
    return result

def compute_color_features(img_rgb, mask):
    zero = {}
    for ch in ['r', 'g', 'b']:
        for stat in ['min', 'max', 'mean', 'median', 'std']:
            zero[f'{ch}_{stat}'] = 0
    if mask is None or mask.sum() == 0:
        return zero
    features = {}
    for i, name in enumerate(['r', 'g', 'b']):
        channel = img_rgb[:, :, i][mask > 0]
        if len(channel) == 0:
            return zero
        features[f'{name}_min'] = float(channel.min())
        features[f'{name}_max'] = float(channel.max())
        features[f'{name}_mean'] = round(float(channel.mean()), 4)
        features[f'{name}_median'] = round(float(np.median(channel)), 4)
        features[f'{name}_std'] = round(float(channel.std()), 4)
    return features

def compute_pixel_ratio(mask):
    if mask is None or mask.sum() == 0:
        return 0.0
    return round(mask.sum() / mask.size, 6)

def compute_custom_features(img_rgb, mask):
    default = {'bg_brightness': 0, 'color_dominance': 0, 'edge_density': 0, 'texture_contrast': 0, 'brightness_mean': 0, 'saturation_mean': 0}
    if mask is None or mask.sum() == 0:
        return default
    outside = (mask == 0)
    bg_pixels = img_rgb[outside]
    bg_brightness = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 0
    inside = (mask > 0)
    r_mean = img_rgb[:, :, 0][inside].mean()
    g_mean = img_rgb[:, :, 1][inside].mean()
    b_mean = img_rgb[:, :, 2][inside].mean()
    total = r_mean + g_mean + b_mean
    color_dominance = round(max(r_mean, g_mean, b_mean) / total, 6) if total > 0 else 0
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_copy = gray.copy()
    gray_copy[mask == 0] = 0
    edges = cv2.Canny(gray_copy.astype(np.uint8), 50, 150)
    mask_pixels = mask.sum()
    edge_density = round((edges > 0).sum() / mask_pixels, 6) if mask_pixels > 0 else 0
    texture_contrast = round(float(gray[inside].std()), 4) if mask_pixels > 0 else 0
    brightness_mean = round(float(np.mean(img_rgb[inside])), 4) if mask_pixels > 0 else 0
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation_mean = round(float(hsv[:, :, 1][inside].mean()), 4) if mask_pixels > 0 else 0
    return {'bg_brightness': round(bg_brightness, 4), 'color_dominance': color_dominance, 'edge_density': edge_density, 'texture_contrast': texture_contrast, 'brightness_mean': brightness_mean, 'saturation_mean': saturation_mean}

def extract_features_for_image(img_id, img_rgb, mask):
    features = {'ID': img_id}
    features.update(compute_symmetry_features(mask))
    features.update(compute_shape_features(mask))
    features.update(compute_color_features(img_rgb, mask))
    features['pixel_ratio'] = compute_pixel_ratio(mask)
    features.update(compute_custom_features(img_rgb, mask))
    return features


def main():
    test_dir = r'C:\Users\13968\Documents\ML机器学习\test'
    model_path = r'C:\Users\13968\Documents\ML机器学习\best_model.pkl'
    output_path = r'C:\Users\13968\Documents\ML机器学习\predictions_251_347.csv'
    
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    if not os.path.exists(test_dir):
        print(f'Error: Test directory not found: {test_dir}')
        print('Please create a test/ folder with images 251.JPG to 347.JPG and masks/ subfolder.')
        return
    
    if not os.path.exists(model_path):
        print(f'Error: Model not found: {model_path}')
        print('Run improved_training.py first to generate best_model.pkl')
        return
    
    print(f'Loading model from {model_path}...')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    le = model_data['label_encoder']
    feature_cols = model_data.get('feature_cols', [])
    model_name = model_data.get('model_name', 'Unknown')
    print(f'Model: {model_name}')
    print(f'Classes: {le.classes_}')
    
    import glob
    test_images = sorted(glob.glob(os.path.join(test_dir, '*.JPG')))
    if not test_images:
        print(f'No JPG images found in {test_dir}')
        return
    
    test_ids = []
    for img_path in test_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        try:
            test_ids.append(int(basename))
        except ValueError:
            continue
    test_ids = sorted(test_ids)
    print(f'Found {len(test_ids)} test images')
    
    all_features = []
    for img_id in test_ids:
        try:
            img_rgb, mask = load_image_and_mask(test_dir, img_id)
            features = extract_features_for_image(img_id, img_rgb, mask)
            all_features.append(features)
            if img_id % 10 == 0 or img_id == test_ids[0]:
                print(f'Processed image {img_id}')
        except Exception as e:
            print(f'Error on image {img_id}: {e}')
    
    features_df = pd.DataFrame(all_features)
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    X_test = features_df[feature_cols].values.astype(np.float64)
    
    X_scaled = scaler.transform(X_test)
    y_pred_encoded = model.predict(X_scaled)
    y_pred_labels = le.inverse_transform(y_pred_encoded)
    
    output_df = pd.DataFrame({
        'ID': features_df['ID'],
        'bug type': y_pred_labels
    })
    output_df.to_csv(output_path, index=False)
    
    print(f'\nPredictions saved to {output_path}')
    print(f'Prediction distribution:')
    print(output_df['bug type'].value_counts().to_string())
    return output_df

if __name__ == '__main__':
    import sys
    import pickle
    import glob
    df = main()
