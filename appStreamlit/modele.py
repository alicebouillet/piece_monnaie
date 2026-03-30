import os
import cv2
import numpy as np
from skimage import transform
from skimage.transform import rotate
from skimage.draw import disk

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

SIZE = 224
MASK_RATIO = 0.70

# -------------------------
# MODELE
# -------------------------
EXTRACTOR = None

def get_extractor():
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    return EXTRACTOR

# -------------------------
# DETECTION PIECE
# -------------------------
def detect_coin_opencv(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = blurred.shape
    min_r = int(min(h, w) * 0.25)
    max_r = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.4,
        param1=80,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None

def apply_mask(image, cx, cy, radius):
    h, w = image.shape[:2]
    national_r = int(radius * MASK_RATIO)

    mask = np.zeros((h, w), dtype=bool)
    rr, cc = disk((cy, cx), national_r, shape=(h, w))
    mask[rr, cc] = True

    masked = image.copy()
    masked[~mask] = 0.5
    return masked

def load_and_mask(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire : {image_path}")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    result = detect_coin_opencv(gray)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    h, w = img_rgb.shape[:2]

    if result is not None:
        cx, cy, radius = result
    else:
        cx, cy = w // 2, h // 2
        radius = int(min(h, w) * 0.42)

    masked = apply_mask(img_rgb, cx, cy, radius)
    resized = transform.resize(masked, (SIZE, SIZE), anti_aliasing=True)
    return resized

# -------------------------
# FEATURES
# -------------------------
def augment(image, n=10):
    rng = np.random.default_rng(42)
    variants = [image]

    for _ in range(n - 1):
        img = rotate(image, rng.uniform(-30, 30), mode="reflect")
        img = np.clip(img * rng.uniform(0.8, 1.2), 0, 1)
        variants.append(img)

    return variants

def extract_batch(images):
    model = get_extractor()
    batch = (np.stack(images) * 255).astype(np.float32)
    feats = model.predict(preprocess_input(batch), verbose=0)

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms > 0, norms, 1)

def extract_features(image_path):
    img = load_and_mask(image_path)
    return extract_batch(augment(img))

# -------------------------
# DATABASE
# -------------------------
DB = None

def load_database(folder="appStreamlit/dataset_image"):
    db = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            feats = extract_features(os.path.join(folder, f))
            db.append((f, feats))
    return db

def init_model():
    global DB
    if DB is None:
        print("Chargement base...")
        DB = load_database("appStreamlit/dataset_image")
    return DB

# -------------------------
# MATCH
# -------------------------
def find_match(test_feats, db):
    best_name = None
    best_score = -1

    for name, db_feats in db:
        sim = (db_feats @ test_feats.T).max()
        if sim > best_score:
            best_score = sim
            best_name = name

    return best_name, float(best_score)

# -------------------------
# PREDICT STREAMLIT
# -------------------------
def predict_streamlit(image_path):
    db = init_model()
    test_feats = extract_features(image_path)
    name, score = find_match(test_feats, db)

    name = name.lower().replace(".jpg", "").replace(".png", "")
    return name, score