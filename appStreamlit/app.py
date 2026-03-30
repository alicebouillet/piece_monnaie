import streamlit as st
import pandas as pd
from PIL import Image
import os
import pydeck as pdk
import tempfile
from modele import predict_streamlit

st.set_page_config(page_title="Collection Euro", layout="wide")

# Mapping noms français -> codes pays ISO2
COUNTRY_CODE_FR = {
    "France": "FR",
    "Allemagne": "DE",
    "Belgique": "BE",
    "Espagne": "ES",
    "Italie": "IT",
    "Portugal": "PT",
    "Pays-Bas": "NL",
    "Autriche": "AT",
    "Finlande": "FI",
    "Grèce": "GR",
    "Irlande": "IE",
    "Luxembourg": "LU",
    "Malte": "MT",
    "Slovaquie": "SK",
    "Slovénie": "SI",
    "Estonie": "EE",
    "Lettonie": "LV",
    "Lituanie": "LT",
    "Chypre": "CY",
    "Croatie": "HR",
    "Andorre": "AD",
    "Monaco": "MC",
    "Saint-Marin": "SM",
    "Vatican": "VA"
}


def country_code_to_flag(code: str) -> str:
    """Convertit un code ISO2 en emoji drapeau."""
    if not code or len(code) != 2:
        return "🇪🇺"
    base = 127397
    try:
        return chr(ord(code[0].upper()) + base) + chr(ord(code[1].upper()) + base)
    except Exception:
        return "🇪🇺"


def country_name_to_flag(name: str) -> str:
    code = COUNTRY_CODE_FR.get(name)
    if code:
        return country_code_to_flag(code)
    return "🇪🇺"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    return pd.read_csv("appStreamlit/data/coins.csv")

df = load_data()

st.title("💰 Ma collection de pièces euro")

# -------------------------
# IA
# -------------------------
st.subheader("🤖 Reconnaissance automatique")

uploaded_file = st.file_uploader("Upload une pièce", type=["jpg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.image(uploaded_file, width=200)

    prediction, score = predict_streamlit(temp_path)

    st.success(f"🪙 {prediction}")
    st.write(f"Score : {score:.4f}")

    if score < 0.5:
        st.warning("⚠️ Prédiction incertaine")

    if st.button("➕ Ajouter à ma collection"):
        mask = df["image"].str.contains(prediction)
        df.loc[mask, "possedee"] = 1
        df.to_csv("appStreamlit/data/coins.csv", index=False)
        st.success("Ajoutée !")

# -------------------------
# Filtres
# -------------------------
pays = st.multiselect("Choisir un pays", df["pays"].unique())
valeurs = st.multiselect("Choisir une valeur", df["valeur"].unique())

filtre = df.copy()

if pays:
    filtre = filtre[filtre["pays"].isin(pays)]

if valeurs:
    filtre = filtre[filtre["valeur"].isin(valeurs)]

# -------------------------
# Toggle pièces possédées
# -------------------------
option = st.radio(
    "Afficher :",
    ["Toutes", "Possédées", "Manquantes"]
)

if option == "Possédées":
    filtre = filtre[filtre["possedee"] == 1]
elif option == "Manquantes":
    filtre = filtre[filtre["possedee"] == 0]
# -------------------------
# AFFICHAGE CLASSIQUE
# -------------------------
st.subheader("🪙 Collection")
for pays_nom in sorted(filtre["pays"].unique()):
    
    st.subheader(f"{country_name_to_flag(pays_nom)} {pays_nom}")
    
    df_pays = filtre[filtre["pays"] == pays_nom]
    
    cols = st.columns(4)  # 4 pièces par ligne
    
    for i, row in df_pays.iterrows():
        col = cols[i % 4]
        
        with col:
            image_path = os.path.join(BASE_DIR, row["image"])

            if os.path.exists(image_path):
                img = Image.open(image_path)
                st.image(img, width=120)
            else:
                st.error("❌ Image manquante")

            st.write(f"💰 {row['valeur']}")
            
            if row["possedee"] == 1:
                st.success("✅")
            else:
                st.warning("❌")