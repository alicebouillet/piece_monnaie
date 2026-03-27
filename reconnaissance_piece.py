import os
import numpy as np
from skimage import io, color, transform
from skimage.feature import ORB, match_descriptors

# -------------------------
# 1. Prétraitement simple
# -------------------------
def preprocess(image_path):
    image = io.imread(image_path)
    
    # gris
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # resize
    image = transform.resize(image, (200, 200))
    
    return image

# -------------------------
# 2. ORB features
# -------------------------
def get_features(image):
    orb = ORB(n_keypoints=300)
    orb.detect_and_extract(image)
    return orb.descriptors

# -------------------------
# 3. Charger base
# -------------------------
def load_database(folder):
    db = []
    
    for file in os.listdir(folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(folder, file)
            
            img = preprocess(path)
            desc = get_features(img)
            
            db.append((file, desc))
    
    return db

# -------------------------
# 4. Comparaison
# -------------------------
def find_match(test_desc, db):
    best_name = None
    best_score = 0
    
    for name, desc in db:
        matches = match_descriptors(test_desc, desc, cross_check=True)
        score = len(matches)
        
        # 👇 AJOUT ICI
        print(name, score)
        
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, best_score

# -------------------------
# 5. Prédiction
# -------------------------
def predict(image_path, db):
    img = preprocess(image_path)
    desc = get_features(img)
    
    return find_match(desc, db)

# -------------------------
# MAIN
# -------------------------


db = load_database("dataset")

result, score = predict("test/test_grece.jpg", db)

print("Résultat :", result)
print("Score :", score)