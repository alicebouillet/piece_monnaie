"""
Reconnaissance de pièces de 2€ — Few-Shot EfficientNetB0
Détection de cercle : OpenCV HoughCircles (robuste fond complexe + taille variable)

Ajustez MASK_RATIO :
  0.40 = motif central serré
  0.50 = défaut recommandé
  0.65 = presque toute la pièce

Installation :
    pip install tensorflow opencv-python scikit-image
"""

import os
import cv2
import numpy as np
from skimage import io, color, transform
from skimage.transform import rotate
from skimage.draw import disk

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

SIZE = 224
MASK_RATIO = 0.70  # fraction du rayon de la pièce à garder

# ─────────────────────────────────────────
# 1. Modèle
# ─────────────────────────────────────────
EXTRACTOR = None

def get_extractor():
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
        print(f"EfficientNetB0 chargé — {EXTRACTOR.output_shape[-1]}D features")
    return EXTRACTOR

# ─────────────────────────────────────────
# 2. Détection OpenCV + masque
# ─────────────────────────────────────────
def detect_coin_opencv(image_uint8_gray):
    """
    Détecte le cercle principal avec cv2.HoughCircles.
    image_uint8_gray : image en niveaux de gris uint8, taille quelconque.
    Retourne (cx, cy, radius) en pixels, ou None si rien trouvé.
    """
    # Flou pour réduire le bruit avant Hough
    blurred = cv2.GaussianBlur(image_uint8_gray, (9, 9), 2)

    h, w = blurred.shape
    min_r = int(min(h, w) * 0.25)
    max_r = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,          # résolution accumulateur
        minDist=min(h, w) * 0.4,   # 1 seul cercle attendu
        param1=80,       # seuil Canny haut
        param2=35,       # seuil accumulateur (baisser si rien trouvé)
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    return None


def apply_mask(image_rgb_float, cx, cy, radius, ratio=MASK_RATIO):
    """
    Applique un masque circulaire centré sur (cx, cy).
    Tout ce qui est hors du cercle national devient gris neutre.
    image_rgb_float : float 0-1, shape (H, W, 3)
    """
    h, w = image_rgb_float.shape[:2]
    national_r = int(radius * ratio)
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = disk((cy, cx), national_r, shape=(h, w))
    mask[rr, cc] = True
    masked = image_rgb_float.copy()
    masked[~mask] = 0.5
    return masked


def load_and_mask(image_path, max_size=800):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Impossible de lire : {image_path}")

    # Redimensionnement préalable
    h, w = img_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    result = detect_coin_opencv(gray)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    h, w = img_rgb.shape[:2]

    if result is not None:
        cx, cy, radius = result
        print(f"    Hough OK  → centre=({cx},{cy}) rayon={radius}px")
    else:
        cx, cy = w // 2, h // 2
        radius = int(min(h, w) * 0.42)
        print(f"    Hough KO  → fallback centre=({cx},{cy}) rayon={radius}px")

    masked = apply_mask(img_rgb, cx, cy, radius)
    resized = transform.resize(masked, (SIZE, SIZE), anti_aliasing=True)
    return resized


# ─────────────────────────────────────────
# 3. Augmentations
# ─────────────────────────────────────────
def augment(image_rgb, n=20, seed=42):
    rng = np.random.default_rng(seed)
    variants = [image_rgb]
    print(f"Augmentation: image initiale ajoutée ({len(variants)})")

    for i in range(n - 1):
        img = image_rgb.copy()
        img = rotate(img, rng.uniform(-30, 30), mode="reflect")
        img = np.clip(img * rng.uniform(0.8, 1.2), 0, 1)
        variants.append(img)
    return variants


# ─────────────────────────────────────────
# 4. Extraction features
# ─────────────────────────────────────────
def extract_batch(images_rgb):
    model = get_extractor()
    batch = (np.stack(images_rgb, axis=0) * 255).clip(0, 255).astype(np.float32)
    feats = model.predict(preprocess_input(batch), verbose=0)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms > 0, norms, 1)


def extract_features(image_path, n_augments=20):
    img = load_and_mask(image_path)
    return extract_batch(augment(img, n=n_augments))


# ─────────────────────────────────────────
# 5. Base de données
# ─────────────────────────────────────────
def load_database(folder, n_augments=20):
    db = []
    extensions = (".png", ".jpg", ".jpeg", ".webp", ".JPG")
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]

    print(f"\nChargement de {len(files)} images × {n_augments} augmentations...")
    for file in files:
        try:
            feats = extract_features(os.path.join(folder, file), n_augments)
            db.append((file, feats))
            print(f"  ✓ {file}")
        except Exception as e:
            print(f"  ✗ {file} — {e}")
    return db


# ─────────────────────────────────────────
# 6. Vote majoritaire
# ─────────────────────────────────────────
def find_match(test_feats, db, top_k=3):
    vote_counts = {name: 0 for name, _ in db}
    score_sums  = {name: 0.0 for name, _ in db}

    for test_vec in test_feats:
        best_name, best_sim = None, -1
        for name, db_feats in db:
            sim = float((db_feats @ test_vec).max())
            score_sums[name] += sim
            if sim > best_sim:
                best_sim, best_name = sim, name
        vote_counts[best_name] += 1

    n = len(test_feats)
    combined = {
        name: 0.6 * (score_sums[name] / n) + 0.4 * (vote_counts[name] / n)
        for name in vote_counts
    }
    ranking = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    print("\n-- Top résultats --")
    for name, score in ranking[:top_k]:
        bar = "█" * int(score * 20)
        print(f"  {name:35s}  score={score:.4f}  votes={vote_counts[name]:2d}/{n}  {bar}")

    return ranking[0]


# ─────────────────────────────────────────
# 7. Prédiction
# ─────────────────────────────────────────
def predict(image_path, db, n_augments=20):
    print(f"\nTest : {image_path}")
    return find_match(extract_features(image_path, n_augments), db)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    db = load_database("dataset", n_augments=20)
    result, score = predict("test/IMG_5415.jpeg", db)
    print(f"\n Résultat : {result}  (score={score:.4f})")