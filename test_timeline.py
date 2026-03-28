
    # ...existing code...
import os
import time
import numpy as np
from skimage import io, color, transform, exposure, filters
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import ORB, match_descriptors, canny
import matplotlib.pyplot as plt
# ...existing code...

# Timeline simple pour suivre l'avancement
class Timeline:
    def __init__(self, total=None):
        self.start = time.time()
        self.events = []
        self.total = total
        self.count = 0

    def add(self, message):
        self.count += 1
        t = time.time() - self.start
        pct = f" ({self.count}/{self.total})" if self.total else ""
        print(f"[{t:.1f}s]{pct} TIMELINE: {message}")
        self.events.append((t, message))

    def summary(self):
        print("\nTIMELINE SUMMARY:")
        for t, msg in self.events:
            print(f"{t:.1f}s - {msg}")
# ...existing code...

def detect_and_crop_circle(image):
# Détection des cercles avec la transformée de Hough
    edges = filters.sobel(image)
    hough_radii = np.arange(80, 110, 2)  # Rayon attendu pour une pièce de 2€ (à ajuster)
    hough_res = feature.hough_circle(edges, hough_radii)

    # Trouver le cercle le plus probable
    accums, cx, cy, radius = feature.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    if accums.size == 0:
        print("Aucun cercle détecté, recadrage central par défaut.")
        return image  # Retourne l'image non recadrée si aucun cercle n'est trouvé

    # Recadrer l'image autour du cercle détecté
    center = (int(cy[0]), int(cx[0]))
    radius = int(radius[0])
    start_x, end_x = center[1] - radius, center[1] + radius
    start_y, end_y = center[0] - radius, center[0] + radius

    # Vérifier les limites
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(image.shape[1], end_x)
    end_y = min(image.shape[0], end_y)

    cropped = image[start_y:end_y, start_x:end_x]
    return cropped
# ...existing code...

def preprocess_center(image_path, show=False, timeline=None):
    if timeline:
        timeline.add(f"Prétraitement démarré: {os.path.basename(image_path)}")
    image = io.imread(image_path)
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Détection et recadrage du cercle
    image = detect_and_crop_circle(image)

    # Redimensionnement
    image = transform.resize(image, (200, 200))

    # Normalisation d'histogramme
    image = exposure.equalize_hist(image)

    # Filtrage pour accentuer les contours
    image = filters.sobel(image)

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title("Image prétraitée")
        plt.axis('off')
        plt.show()

    if timeline:
        timeline.add(f"Prétraitement terminé: {os.path.basename(image_path)}")
    return image

def get_features(image, timeline=None):
    if timeline:
        timeline.add("Extraction features ORB démarrée")
    orb = ORB(n_keypoints=500)
    orb.detect_and_extract(image)
    keypoints = orb.keypoints
    descriptors = orb.descriptors
    if timeline:
        timeline.add(f"Extraction features ORB terminée ({len(keypoints)} keypoints)")
    return descriptors, keypoints

def show_keypoints(image, keypoints):
    # ...existing code...
    plt.show()

def show_matches(img1, img2, keypoints1, keypoints2, matches):
    # ...existing code...
    plt.show()

def load_database(folder, timeline=None):
    if timeline:
        timeline.add(f"Chargement DB démarré: {folder}")
    db = []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    total = len(files)
    for i, file in enumerate(files, 1):
        if timeline:
            timeline.add(f"DB: traitement {file} ({i}/{total})")
        path = os.path.join(folder, file)
        img = preprocess_center(path, timeline=timeline)
        desc, keypoints = get_features(img, timeline=timeline)
        db.append((file, desc, img, keypoints))

    if timeline:
        timeline.add(f"Chargement DB terminé: {len(db)} images")
    return db

def find_match(test_desc, db, test_img, test_keypoints, show_matches=False, min_score=30, timeline=None):
    if timeline:
        timeline.add("Recherche de match démarrée")
    best_name = None
    best_score = 0

    for entry in db:
        name, desc, img, keypoints = entry
        if desc is None or test_desc is None:
            continue

        if timeline:
            timeline.add(f"Matching avec {name}")

        matches = match_descriptors(test_desc, desc, cross_check=True)
        score = len(matches)

        if len(desc) > 0:
            score = score / len(desc) * 100

        print(f"{name}: {score:.1f}% matches")

        if show_matches and score > 0:
            show_matches(test_img, img, test_keypoints, keypoints, matches)

        if score > best_score and score > min_score:
            best_score = score
            best_name = name

    if timeline:
        timeline.add(f"Recherche de match terminée, meilleur: {best_name} ({best_score:.1f}%)")
    return best_name, best_score

def predict(image_path, db, show_matches=False, timeline=None):
    if timeline:
        timeline.add(f"Prédiction démarrée: {os.path.basename(image_path)}")
    img = preprocess_center(image_path, timeline=timeline)
    desc, keypoints = get_features(img, timeline=timeline)
    result = find_match(desc, db, img, keypoints, show_matches, timeline=timeline)
    if timeline:
        timeline.add(f"Prédiction terminée: {os.path.basename(image_path)}")
    return result

if __name__ == "__main__":
    timeline = Timeline()
    db = load_database("dataset", timeline=timeline)
    result, score = predict("test/IMG_5415.jpg", db, show_matches=True, timeline=timeline)
    print(f"Résultat : {result}")
    print(f"Score : {score:.1f}%")
    timeline.summary()
# ...existing code...