from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import base64
import os
import modele

app = Flask(__name__)
CORS(app)
DB = None

FLAG_MAP = {
    'france': '🇫🇷',
    'allemagne': '🇩🇪',
    'grece': '🇬🇷',
    'autriche': '🇦🇹',
    'andorre': '🇦🇩',
    'luxembourg': '🇱🇺',
    'saint-marin': '🇸🇲',
}

def ensure_db():
    global DB
    if DB is None:
        DB = modele.load_database('dataset', n_augments=20)
    return DB

@app.route('/identify', methods=['POST'])
def identify():
    data = request.get_json() or {}
    imgdata = data.get('imageData') or data.get('imageBase64')
    if not imgdata:
        return jsonify({'error': 'missing imageData'}), 400

    # handle data:...;base64,... format
    if imgdata.startswith('data:'):
        imgdata = imgdata.split(',', 1)[1]
    try:
        b = base64.b64decode(imgdata)
    except Exception as e:
        return jsonify({'error': 'invalid base64'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
        f.write(b)
        tmp_path = f.name

    try:
        db = ensure_db()
        name, score = modele.predict(tmp_path, db, n_augments=20)
        # name is filename from dataset
        base = os.path.splitext(os.path.basename(name))[0]
        parts = base.split('_')
        country = parts[0].capitalize()
        year = ''
        for p in parts:
            if p.isdigit() and len(p) == 4:
                year = p
        denom = '2 €' if '2' in base else ("1 cts" if '1cts' in base or '1cts' in name else '—')
        flag = FLAG_MAP.get(parts[0].lower(), '🪙')
        confidence = 'haute' if score > 0.6 else ('moyenne' if score > 0.4 else 'faible')
        result = {
            'country': country,
            'flag': flag,
            'year': year,
            'denomination': denom,
            'currency': 'Euro',
            'condition': '—',
            'description': f"Match probable: {base}",
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
