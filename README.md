
# Collection de pièces

Petite application web statique pour identifier et collectionner des pièces via un modèle d'IA (Mistral). Interface mobile-first, stockage local pour la clé API et la collection.

**Fonctionnalités**
- Prendre / importer une photo de pièce
- Envoyer la photo à un modèle IA pour identification
- Enregistrer les pièces identifiées dans une collection locale
- Gestion simple de la clé API et confidentialité

**Installation & test local**
Aucun build nécessaire — c'est une app statique. Ouvrez `index.html` dans un navigateur ou servez le dossier localement :

```bash
# depuis le dossier project
python -m http.server 8000
# puis visiter http://localhost:8000
```

**Structure du projet**
- `index.html` — page principale
- `css/style.css` — styles
- `js/scripts.js` — logique front-end
- `manifest.json` — manifest PWA (si présent)

**Utilisation**
1. Ouvrir l'app dans le navigateur
2. Créer/obtenir une clé API Mistral (console.mistral.ai)
3. Coller la clé dans l'écran d'onboarding
4. Prendre une photo de la pièce et cliquer sur "Identifier"

**Accessibilité & vie privée**
- La clé API est stockée localement (prévoir chiffrement si besoin)
- Seules les photos sont envoyées à Mistral pour analyse (indiqué dans l'UI)
- Améliorations recommandées : labels associées aux inputs, `aria-live` pour les messages, éviter handlers inline

**Contribuer**
Proposez des issues ou forks pour : meilleure accessibilité, gestion sécurisée de la clé, tests, et packaging.

**Licence**
Ajoutez la licence souhaitée (ex. MIT) si vous partagez publiquement.
