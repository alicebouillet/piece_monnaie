import os
import numpy as np
from skimage import io, color, transform, exposure, filters
from skimage.feature import ORB, match_descriptors, hog
from skimage.draw import disk
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

# ─────────────────────────────────────────
# 1. Détection automatique de la pièce
#    via Hough Circle Transform
# ─────────────────────────────────────────
def detect_coin_circle(image_gray, size=200):
    """
    Détecte le cercle de la pièce dans l'image.
    Retourne (cy, cx, radius) en coordonnées de l'image redimensionnée.
    Fallback : centre de l'image, rayon = 45% de la taille.
    """
    img = transform.resize(image_gray, (size, size), anti_aliasing=True)

    # Contours Canny
    edges = canny(img, sigma=2.0, low_threshold=0.1, high_threshold=0.3)

    # Chercher des cercles entre 35% et 50% de la taille de l'image
    min_r = int(size * 0.35)
    max_r = int(size * 0.50)
    radii = np.arange(min_r, max_r, 2)

    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii_found = hough_circle_peaks(
        hough_res, radii, total_num_peaks=1, min_xdistance=20, min_ydistance=20
    )

    if len(cx) > 0:
        return int(cy[0]), int(cx[0]), int(radii_found[0])
    else:
        # Fallback : pièce supposée centrée
        return size // 2, size // 2, int(size * 0.45)


# ─────────────────────────────────────────
# 2. Masque : zone nationale centrale
#    (55% intérieur du rayon détecté)
# ─────────────────────────────────────────
def apply_national_mask(image, cy, cx, coin_radius, national_ratio=0.55):
    """
    Garde uniquement le disque national (centre de la pièce).
    national_ratio : fraction du rayon de la pièce à conserver.
    """
    national_radius = int(coin_radius * national_ratio)
    mask = np.zeros(image.shape, dtype=bool)
    rr, cc = disk((cy, cx), national_radius, shape=image.shape)
    mask[rr, cc] = True
    masked = image.copy()
    masked[~mask] = 0
    return masked


# ─────────────────────────────────────────
# 3. Prétraitement complet
# ─────────────────────────────────────────
def preprocess(image_path, size=200):
    image = io.imread(image_path)

    if image.ndim == 3 and image.shape[2] == 4:
        image = color.rgba2rgb(image)
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Normalisation contraste
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    image = transform.resize(image, (size, size), anti_aliasing=True)

    # Détection automatique de la pièce
    cy, cx, radius = detect_coin_circle(image, size=size)

    # Masque sur le motif national uniquement
    image = apply_national_mask(image, cy, cx, radius, national_ratio=0.55)

    return image


# ─────────────────────────────────────────
# 4. Descripteurs ORB + HOG
# ─────────────────────────────────────────
def get_orb_descriptors(image):
    orb = ORB(n_keypoints=300)
    try:
        orb.detect_and_extract(image)
        return orb.descriptors
    except Exception:
        return None


def get_hog_descriptor(image):
    desc, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True
    )
    norm = np.linalg.norm(desc)
    return desc / norm if norm > 0 else desc


def get_features(image):
    return {
        "orb": get_orb_descriptors(image),
        "hog": get_hog_descriptor(image),
    }


# ─────────────────────────────────────────
# 5. Base de données
# ─────────────────────────────────────────
def load_database(folder):
    db = []
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(extensions)]
    print(f"Chargement de {len(files)} images...")
    for file in files:
        path = os.path.join(folder, file)
        img = preprocess(path)
        features = get_features(img)
        db.append((file, features))
        print(f"  ✓ {file}")
    return db


# ─────────────────────────────────────────
# 6. Score combiné ORB + HOG
# ─────────────────────────────────────────
def compute_score(feat_test, feat_db, w_orb=0.4, w_hog=0.6):
    score_orb = 0
    if feat_test["orb"] is not None and feat_db["orb"] is not None:
        matches = match_descriptors(feat_test["orb"], feat_db["orb"], cross_check=True)
        score_orb = min(len(matches) / 100.0, 1.0)

    score_hog = float(np.dot(feat_test["hog"], feat_db["hog"]))

    return w_orb * score_orb + w_hog * score_hog


# ─────────────────────────────────────────
# 7. Recherche
# ─────────────────────────────────────────
def find_match(test_features, db, top_k=3):
    scores = [(name, compute_score(test_features, feat)) for name, feat in db]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n── Top résultats ──")
    for name, score in scores[:top_k]:
        print(f"  {name:30s}  score={score:.4f}")

    return scores[0]


# ─────────────────────────────────────────
# 8. Prédiction
# ─────────────────────────────────────────
def predict(image_path, db):
    print(f"\nTest : {image_path}")
    img = preprocess(image_path)
    features = get_features(img)
    return find_match(features, db)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    db = load_database("dataset_aug")
    result, score = predict("test/IMG_5415.jpg", db)
    print(f"\n Résultat : {result}  (score={score:.4f})")