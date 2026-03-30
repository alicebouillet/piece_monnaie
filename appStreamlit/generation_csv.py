import pandas as pd

pays_coords = {
    "France": (46.6034, 1.8883),
    "Allemagne": (51.1657, 10.4515),
    "Belgique": (50.5039, 4.4699),
    "Espagne": (40.4637, -3.7492),
    "Italie": (41.8719, 12.5674),
    "Portugal": (39.3999, -8.2245),
    "Pays-Bas": (52.1326, 5.2913),
    "Autriche": (47.5162, 14.5501),
    "Finlande": (61.9241, 25.7482),
    "Grèce": (39.0742, 21.8243),
    "Irlande": (53.1424, -7.6921),
    "Luxembourg": (49.8153, 6.1296),
    "Malte": (35.9375, 14.3754),
    "Slovaquie": (48.6690, 19.6990),
    "Slovénie": (46.1512, 14.9955),
    "Estonie": (58.5953, 25.0136),
    "Lettonie": (56.8796, 24.6032),
    "Lituanie": (55.1694, 23.8813),
    "Chypre": (35.1264, 33.4299),
    "Croatie": (45.1, 15.2),
    "Andorre": (42.5063, 1.5218),
    "Monaco": (43.7384, 7.4246),
    "Saint-Marin": (43.9424, 12.4578),
    "Vatican": (41.9029, 12.4534),
}

valeurs = ["1c","2c","5c","10c","20c","50c","1€","2€"]

rows = []

for pays, (lat, lon) in pays_coords.items():
    for val in valeurs:
        nom_image = f"images/{pays.lower().replace(' ', '_')}_{val.replace('€','e')}.jpg"
        
        rows.append({
            "pays": pays,
            "valeur": val,
            "image": nom_image,
            "possedee": 0,
            "lat": lat,
            "lon": lon
        })

df = pd.DataFrame(rows)
df.to_csv("coins.csv", index=False)

print("CSV généré !")