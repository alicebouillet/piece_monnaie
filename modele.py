"""
Reconnaissance de pièces de 2€ — Few-Shot avec EfficientNetB0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Principe : EfficientNet extrait un vecteur de 1280 valeurs par image.
On compare ces vecteurs par similarité cosinus — aucun entraînement
nécessaire, fonctionne avec 1 seule image par pays.

Installation :
    pip install tensorflow scikit-image opencv-python
"""

import os
import numpy as np
from skimage import io, color, transform, exposure
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import disk

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model

# ─────────────────────────────────────────
# 1. Modèle EfficientNetB0 — extracteur
# ─────────────────────────────────────────
def build_extractor():
    """
    EfficientNetB0 pré-entraîné sur ImageNet, sans la tête de classification.
    Retourne un vecteur de 1280 dimensions par image.
    """
    base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    print(f"Modèle chargé — {base.output_shape[-1]}D features")
    return base

EXTRACTOR = None  # chargé une seule fois

def get_extractor():
    global EXTRACTOR
    if EXTRACTOR is None:
        EXTRACTOR = build_extractor()
    return EXTRACTOR


# ─────────────────────────────────────────
# 2. Détection automatique de la pièce
# ─────────────────────────────────────────
def detect_coin_circle(image_gray, size=224):
    """
    Détecte le cercle de la pièce via Hough.
    Fallback : centre de l'image, rayon = 45%.
    """
    edges = canny(image_gray, sigma=2.0, low_threshold=0.1, high_threshold=0.3)
    min_r, max_r = int(size * 0.35), int(size * 0.50)
    radii = np.arange(min_r, max_r, 2)
    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii_found = hough_circle_peaks(
        hough_res, radii, total_num_peaks=1, min_xdistance=20, min_ydistance=20
    )
    if len(cx) > 0:
        return int(cy[0]), int(cx[0]), int(radii_found[0])
    return size // 2, size // 2, int(size * 0.45)


def apply_national_mask(image_rgb, size=224, national_ratio=0.55):
    """
    Masque circulaire sur le motif national (centre de la pièce).
    Travaille sur une image RGB — le fond masqué devient gris neutre.
    """
    gray = color.rgb2gray(image_rgb)
    cy, cx, radius = detect_coin_circle(gray, size=size)
    national_r = int(radius * national_ratio)

    mask = np.zeros((size, size), dtype=bool)
    rr, cc = disk((cy, cx), national_r, shape=(size, size))
    mask[rr, cc] = True

    masked = image_rgb.copy()
    masked[~mask] = 0.5   # fond gris neutre (valeur normalisée 0-1)
    return masked


# ─────────────────────────────────────────
# 3. Prétraitement pour EfficientNet
#    EfficientNet attend : RGB, 224×224, float32
# ─────────────────────────────────────────
def preprocess(image_path, size=224, mask=True):
    image = io.imread(image_path)

    # Gérer RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = (color.rgba2rgb(image) * 255).astype(np.uint8)
    elif image.ndim == 2:
        # Niveaux de gris → RGB
        image = (np.stack([image] * 3, axis=-1) * 255).astype(np.uint8)

    # Resize 224×224
    image = transform.resize(image, (size, size), anti_aliasing=True)  # float 0-1

    # Masque national
    if mask:
        image = apply_national_mask(image, size=size)

    # Convertir en uint8 puis appliquer preprocess_input EfficientNet
    image_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    image_batch = np.expand_dims(image_uint8, axis=0).astype(np.float32)
    return preprocess_input(image_batch)   # shape (1, 224, 224, 3)


# ─────────────────────────────────────────
# 4. Extraction de features
# ─────────────────────────────────────────
def extract_features(image_path, mask=True):
    """Retourne un vecteur L2-normalisé de 1280 dimensions."""
    model = get_extractor()
    img = preprocess(image_path, mask=mask)
    features = model.predict(img, verbose=0)[0]   # (1280,)
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


# ─────────────────────────────────────────
# 5. Base de données
# ─────────────────────────────────────────
def load_database(folder, mask=True):
    """
    Charge toutes les images du dossier et extrait leurs features.
    Structure attendue : dataset/france.jpg, dataset/allemagne.jpg, etc.
    """
    db = []
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]

    print(f"\nChargement de {len(files)} images (EfficientNetB0)...")
    for file in files:
        path = os.path.join(folder, file)
        try:
            feat = extract_features(path, mask=mask)
            db.append((file, feat))
            print(f"  ✓ {file}")
        except Exception as e:
            print(f"  ✗ {file} — erreur : {e}")

    return db


# ─────────────────────────────────────────
# 6. Similarité cosinus + recherche
# ─────────────────────────────────────────
def find_match(test_feat, db, top_k=3):
    """
    Compare par similarité cosinus (vecteurs déjà L2-normalisés → dot product).
    Score entre 0 et 1 : plus c'est proche de 1, plus c'est similaire.
    """
    scores = [(name, float(np.dot(test_feat, feat))) for name, feat in db]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n── Top résultats ──")
    for name, score in scores[:top_k]:
        bar = "█" * int(score * 20)
        print(f"  {name:30s}  {score:.4f}  {bar}")

    return scores[0]


# ─────────────────────────────────────────
# 7. Prédiction
# ─────────────────────────────────────────
def predict(image_path, db, mask=True):
    print(f"\nTest : {image_path}")
    feat = extract_features(image_path, mask=mask)
    return find_match(feat, db)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Chargement de la base
    db = load_database("dataset", mask=True)

    # Prédiction
    result, score = predict("test/test_grece.jpg", db)
    print(f"\n Résultat : {result}  (similarité={score:.4f})")

    # ── Astuce debug ──────────────────────────────────────────────
    # Si les résultats sont mauvais, testez sans masque pour comparer :
    #   db_nomask = load_database("dataset", mask=False)
    #   result2, score2 = predict("test/test_grece.jpg", db_nomask, mask=False)
    # ─────────────────────────────────────────────────────────────