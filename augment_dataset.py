import os
import numpy as np
from skimage import io, color, transform, exposure, util
from skimage.filters import gaussian

# ─────────────────────────────────────────
# Augmentations disponibles
# ─────────────────────────────────────────

def aug_rotate(img, angle):
    return transform.rotate(img, angle, mode='edge')

def aug_zoom(img, factor):
    h, w = img.shape[:2]
    zoomed = transform.rescale(img, factor, mode='edge', anti_aliasing=True)
    zh, zw = zoomed.shape[:2]
    # Crop ou pad pour revenir à la taille originale
    if factor > 1:
        y0 = (zh - h) // 2
        x0 = (zw - w) // 2
        return zoomed[y0:y0+h, x0:x0+w]
    else:
        pad_y = (h - zh) // 2
        pad_x = (w - zw) // 2
        out = np.zeros_like(img)
        out[pad_y:pad_y+zh, pad_x:pad_x+zw] = zoomed
        return out

def aug_brightness(img, delta):
    return np.clip(img + delta, 0, 1)

def aug_contrast(img, factor):
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 1)

def aug_noise(img, sigma):
    noisy = img + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 1)

def aug_blur(img, sigma):
    return gaussian(img, sigma=sigma)

def aug_flip_h(img):
    return img[:, ::-1]

def aug_flip_v(img):
    return img[::-1, :]

# ─────────────────────────────────────────
# Pipeline d'augmentation
# ─────────────────────────────────────────

AUGMENTATIONS = [
    ("rot+15",    lambda img: aug_rotate(img, 15)),
    ("rot-15",    lambda img: aug_rotate(img, -15)),
    ("rot+30",    lambda img: aug_rotate(img, 30)),
    ("rot-30",    lambda img: aug_rotate(img, -30)),
    ("rot+45",    lambda img: aug_rotate(img, 45)),
    ("rot+90",    lambda img: aug_rotate(img, 90)),
    ("rot+180",   lambda img: aug_rotate(img, 180)),
    ("zoom_in",   lambda img: aug_zoom(img, 1.15)),
    ("zoom_out",  lambda img: aug_zoom(img, 0.85)),
    ("bright+",   lambda img: aug_brightness(img, 0.15)),
    ("bright-",   lambda img: aug_brightness(img, -0.15)),
    ("contrast+", lambda img: aug_contrast(img, 1.4)),
    ("contrast-", lambda img: aug_contrast(img, 0.7)),
    ("noise_s",   lambda img: aug_noise(img, 0.02)),
    ("noise_m",   lambda img: aug_noise(img, 0.04)),
    ("blur",      lambda img: aug_blur(img, 1.0)),
    ("flip_h",    lambda img: aug_flip_h(img)),
    ("flip_v",    lambda img: aug_flip_v(img)),
    # Combinaisons réalistes
    ("rot15_bn",  lambda img: aug_noise(aug_brightness(aug_rotate(img, 15), 0.1), 0.02)),
    ("rot-15_bd", lambda img: aug_noise(aug_brightness(aug_rotate(img, -15), -0.1), 0.02)),
    ("zoom_br+",  lambda img: aug_brightness(aug_zoom(img, 1.1), 0.1)),
    ("zoom_br-",  lambda img: aug_brightness(aug_zoom(img, 0.9), -0.1)),
    ("rot30_blur",lambda img: aug_blur(aug_rotate(img, 30), 0.8)),
    ("flip_rot",  lambda img: aug_rotate(aug_flip_h(img), 20)),
]

# ─────────────────────────────────────────
# Génération du dataset augmenté
# ─────────────────────────────────────────

def augment_dataset(input_folder, output_folder, keep_original=True):
    os.makedirs(output_folder, exist_ok=True)
    extensions = (".png", ".jpg", ".jpeg", ".webp")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

    print(f"{len(files)} image(s) trouvée(s) → génération de {len(files) * (len(AUGMENTATIONS) + 1)} images...\n")

    for file in files:
        path = os.path.join(input_folder, file)
        image = io.imread(path)

        # Convertir en float [0,1]
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255.0

        # Gérer RGBA
        if image.ndim == 3 and image.shape[2] == 4:
            image = color.rgba2rgb(image)

        # Convertir en niveaux de gris
        if image.ndim == 3:
            image = color.rgb2gray(image)

        name, ext = os.path.splitext(file)

        # Garder l'original
        if keep_original:
            out_path = os.path.join(output_folder, f"{name}_orig.png")
            io.imsave(out_path, (image * 255).astype(np.uint8))

        # Générer les variantes
        for aug_name, aug_fn in AUGMENTATIONS:
            try:
                augmented = aug_fn(image)
                augmented = np.clip(augmented, 0, 1)
                out_path = os.path.join(output_folder, f"{name}_{aug_name}.png")
                io.imsave(out_path, (augmented * 255).astype(np.uint8))
            except Exception as e:
                print(f"  Erreur {aug_name} sur {file} : {e}")

        print(f"  ✓ {file}  →  {len(AUGMENTATIONS) + 1} images générées")

    total = len(os.listdir(output_folder))
    print(f"\nDataset augmenté prêt dans '{output_folder}' ({total} images au total)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    augment_dataset(
        input_folder="dataset",      # vos images originales (1 par pays)
        output_folder="dataset_aug", # dataset augmenté généré ici
        keep_original=True
    )