let collection = JSON.parse(localStorage.getItem('numismaia_v3') || '[]');
let currentResult = null;
let imageBase64 = null;
let imageType = 'image/jpeg';

window.addEventListener('DOMContentLoaded', () => {
  const key = localStorage.getItem('mistral_key');
  if (key) launchApp(key);
});

function toggleKey() {
  const i = document.getElementById('apiKeyInput');
  i.type = i.type === 'password' ? 'text' : 'password';
}

function startApp() {
  const key = document.getElementById('apiKeyInput').value.trim();
  if (!key) { alert('Entre ta clé API Mistral 😊'); return; }
  localStorage.setItem('mistral_key', key);
  launchApp(key);
}

function launchApp(key) {
  document.getElementById('onboarding').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
  updateKeyPreview(key);
  updateStats();
  renderPokedex();
}

function updateKeyPreview(k) {
  document.getElementById('keyPreview').textContent = k.substring(0,8) + '••••••••';
}

function switchTab(name) {
  ['identify','pokedex','settings'].forEach(n => {
    document.getElementById('page-'+n).classList.toggle('active', n===name);
    document.getElementById('nav-'+n).classList.toggle('active', n===name);
  });
  if (name === 'pokedex') renderPokedex();
}

document.getElementById('fileInput').addEventListener('change', function() {
  if (!this.files[0]) return;
  imageType = this.files[0].type || 'image/jpeg';
  const reader = new FileReader();
  reader.onload = e => {
    imageBase64 = e.target.result.split(',')[1];
    const prev = document.getElementById('previewImg');
    prev.src = e.target.result;
    prev.style.display = 'block';
    document.getElementById('uploadEmoji').style.display = 'none';
    document.getElementById('identifyBtn').disabled = false;
    document.getElementById('resultCard').classList.remove('visible');
    document.getElementById('errorPill').classList.remove('visible');
  };
  reader.readAsDataURL(this.files[0]);
});

async function identify() {
  if (!imageBase64) return;
  const key = localStorage.getItem('mistral_key');
  if (!key) { changeKey(); return; }
  document.getElementById('identifyBtn').disabled = true;
  document.getElementById('loadingPills').classList.add('visible');
  document.getElementById('resultCard').classList.remove('visible');
  document.getElementById('errorPill').classList.remove('visible');
  try {
    const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
      body: JSON.stringify({
        model: 'pixtral-12b-2409',
        max_tokens: 800,
        messages: [{
          role: 'user',
          content: [
            { type: 'image_url', image_url: { url: `data:${imageType};base64,${imageBase64}` } },
            { type: 'text', text: `Tu es un expert numismate. Analyse cette pièce et réponds UNIQUEMENT avec un JSON valide sans markdown ni backticks :
{"country":"Nom du pays en français","flag":"Emoji drapeau","year":"Année estimée","denomination":"Valeur faciale ex: 1 Euro","currency":"Devise","condition":"Neuve ou Très bien ou Bien ou Usée","description":"2-3 phrases sur cette pièce","confidence":"haute ou moyenne ou faible"}
Si ce n'est pas une pièce, mets country: "Inconnu".` }
          ]
        }]
      })
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.message || 'Erreur API'); }
    const data = await res.json();
    currentResult = JSON.parse(data.choices[0].message.content.replace(/```json|```/g,'').trim());
    showResult(currentResult);
  } catch(err) {
    const box = document.getElementById('errorPill');
    box.textContent = err.message.includes('401') ? '❌ Clé API invalide — va dans Réglages.' : '❌ ' + err.message;
    box.classList.add('visible');
  } finally {
    document.getElementById('loadingPills').classList.remove('visible');
    document.getElementById('identifyBtn').disabled = false;
  }
}

function isOwned(r) {
  return collection.some(c => c.country===r.country && c.denomination===r.denomination && c.year===r.year);
}

function showResult(r) {
  const owned = isOwned(r);
  document.getElementById('rFlag').textContent = r.flag || '🪙';
  document.getElementById('rCountry').textContent = r.country;
  document.getElementById('rBadge').innerHTML = owned
    ? '<span class="result-badge badge-owned">⭐ Déjà dans ta collection !</span>'
    : '<span class="result-badge badge-new">✨ Nouvelle pièce !</span>';
  document.getElementById('rChips').innerHTML = `
    <div class="chip"><div class="chip-key">Année</div><div class="chip-val">${r.year||'—'}</div></div>
    <div class="chip"><div class="chip-key">Valeur</div><div class="chip-val">${r.denomination||'—'}</div></div>
    <div class="chip"><div class="chip-key">Devise</div><div class="chip-val">${r.currency||'—'}</div></div>
    <div class="chip"><div class="chip-key">État</div><div class="chip-val">${r.condition||'—'}</div></div>
  `;
  document.getElementById('rDesc').textContent = r.description || '';
  const btn = document.getElementById('addBtn');
  btn.disabled = owned;
  btn.textContent = owned ? '⭐ Déjà dans ta collection !' : '✨ Ajouter à ma collection !';
  document.getElementById('resultCard').classList.add('visible');
}

function addToCollection() {
  if (!currentResult || isOwned(currentResult)) return;
  collection.unshift({ ...currentResult, id: Date.now(), addedAt: new Date().toLocaleDateString('fr-FR') });
  save(); updateStats(); showResult(currentResult);
  setTimeout(() => switchTab('pokedex'), 500);
}

function renderPokedex() {
  const q = (document.getElementById('searchInput').value || '').toLowerCase();
  const filtered = collection.filter(c =>
    !q || c.country.toLowerCase().includes(q) ||
    (c.denomination||'').toLowerCase().includes(q) ||
    (c.year||'').includes(q)
  );
  const groups = {};
  filtered.forEach(c => {
    if (!groups[c.country]) groups[c.country] = { flag: c.flag||'🪙', coins: [] };
    groups[c.country].coins.push(c);
  });
  const countries = Object.keys(groups).sort();
  document.getElementById('pokedexCount').textContent = countries.length + ' pays';
  const list = document.getElementById('pokedexList');
  if (!countries.length) {
    list.innerHTML = `<div class="empty-state">
      <div class="empty-emoji">${q?'🔍':'🌍'}</div>
      <div class="empty-msg">${q?'Aucun résultat pour "'+q+'"':'Ton Pokédex est vide !<br>Identifie ta première pièce 🪙'}</div>
    </div>`;
    return;
  }
  list.innerHTML = countries.map(country => {
    const g = groups[country];
    const tiles = g.coins.map(c => `
      <div class="coin-tile">
        <button class="coin-tile-remove" onclick="removeItem(${c.id})">✕</button>
        <div class="coin-tile-flag">${c.flag||'🪙'}</div>
        <div class="coin-tile-denom">${c.denomination||'?'}</div>
        <div class="coin-tile-year">${c.year||'?'}</div>
        <div class="coin-tile-cond">${c.condition||'?'}</div>
      </div>
    `).join('');
    return `<div class="country-group">
      <div class="country-header">
        <span class="country-flag-big">${g.flag}</span>
        <span class="country-name">${country}</span>
        <span class="country-pill">${g.coins.length} pièce${g.coins.length>1?'s':''}</span>
      </div>
      <div class="coins-scroll">${tiles}</div>
    </div>`;
  }).join('');
}

function removeItem(id) {
  if (!confirm('Supprimer cette pièce ?')) return;
  collection = collection.filter(c => c.id !== id);
  save(); renderPokedex(); updateStats();
  if (currentResult) showResult(currentResult);
}

function updateStats() {
  document.getElementById('statTotal').textContent = collection.length;
  document.getElementById('statCountries').textContent = new Set(collection.map(c=>c.country)).size;
}

function save() { localStorage.setItem('numismaia_v3', JSON.stringify(collection)); }

function changeKey() {
  const k = prompt('Nouvelle clé API Mistral :');
  if (k && k.trim()) { localStorage.setItem('mistral_key', k.trim()); updateKeyPreview(k.trim()); alert('✅ Clé mise à jour !'); }
}

function clearCollection() {
  if (!confirm('Supprimer TOUTE ta collection ? 😱')) return;
  collection = []; save(); renderPokedex(); updateStats();
}