import streamlit as st
import pandas as pd
from shapely.geometry import Point
from geopy.distance import geodesic
import joblib
import requests
from io import BytesIO

st.set_page_config(page_title="Prediksi Walk-in POS", layout="centered")
st.title("\U0001F4CD Scoring Lokasi POS")

# ===== INPUT USER =====
total_motor = st.number_input("Masukkan jumlah kendaraan motor", min_value=0, step=10)
total_mobil = st.number_input("Masukkan jumlah kendaraan mobil", min_value=0, step=10)
lebar_jalan = st.number_input("Masukkan lebar jalan (meter)", min_value=0.0, step=0.5)

latitude = st.number_input("Latitude lokasi", value=-6.99705, format="%.6f")
longitude = st.number_input("Longitude lokasi", value=110.34607, format="%.6f")

# ===== LOAD DATA =====
@st.cache_data
def load_csv_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url)
    return pd.read_csv(BytesIO(response.content))

# Gunakan versi hosting yang stabil (Dropbox/HuggingFace/GitHub raw)
data_poi_url = "https://storage.googleapis.com/data_poi_jawa/data_poi_longlat.csv"
df_kelurahan_url = "https://raw.githubusercontent.com/muthia11/ScoringLocation_V2/main/kelurahan_jawa.csv"

data_poi = load_csv_from_url(data_poi_url)
df_kelurahan = load_csv_from_url(df_kelurahan_url)

# ===== CARI KELURAHAN TERDEKAT =====
def cari_kelurahan_terdekat(lat_user, lon_user, df_kelurahan):
    df_kelurahan['jarak'] = df_kelurahan.apply(
        lambda row: geodesic((lat_user, lon_user), (row['Latitude'], row['Longitude'])).kilometers, axis=1
    )
    return df_kelurahan.loc[df_kelurahan['jarak'].idxmin()]

# ===== CEK POI DALAM RADIUS =====
def is_within_radius(poi, center_point, radius_km):
    poi_point = (poi['Latitude'], poi['Longitude'])
    return geodesic(center_point, poi_point).kilometers <= radius_km

center_point = (latitude, longitude)
kelurahan_data = cari_kelurahan_terdekat(latitude, longitude, df_kelurahan)
kelurahan_name = kelurahan_data['DESA_KELUR']
jumlah_pen = kelurahan_data['JUMLAH_PEN']
luas_wil = kelurahan_data['LUAS_WILAY']

filtered_poi_df = data_poi[data_poi.apply(is_within_radius, center_point=center_point, radius_km=2.3, axis=1)]

if filtered_poi_df.empty:
    st.warning("Tidak ada POI dalam radius 2.3 km dari lokasi ini.")
    st.stop()

if 'grouping' not in filtered_poi_df.columns:
    st.error("Kolom 'grouping' tidak ditemukan dalam data POI.")
    st.stop()

poi_categories = ['apartments', 'café', 'community_centre', 'fast_food', 'hospital', 'industrial', 'library',
                  'market_place', 'military', 'office', 'orchard', 'park', 'pharmacy', 'place_of_worship',
                  'residential', 'shop', 'stadium', 'tourism']

grouped_poi_count = filtered_poi_df.groupby('grouping').size().to_dict()
poi_counts = {cat: grouped_poi_count.get(cat, 0) for cat in poi_categories}

density = jumlah_pen / luas_wil if jumlah_pen and luas_wil else None
poi_counts.update({'Density': density, 'Kelurahan': kelurahan_name, 'latitude': latitude, 'longitude': longitude})

# ===== IMPORT MODEL =====

import os

def download_file_from_github(url, local_filename):
    if not os.path.exists(local_filename):
        response = requests.get(url)
        with open(local_filename, 'wb') as f:
            f.write(response.content)

base_url = "https://raw.githubusercontent.com/muthia11/ScoringLocation_V2/main/"

model_files = {
    "scaler_point.pkl": base_url + "scaler_point.pkl",
    "model_point_only.pkl": base_url + "model_point_only.pkl",
    "model_type_motor.pkl": base_url + "model_type_motor.pkl",
    "model_type_mobil.pkl": base_url + "model_type_mobil.pkl",
    "model_dur_motor.pkl": base_url + "model_dur_motor.pkl",
    "model_dur_mobil.pkl": base_url + "model_dur_mobil.pkl",
    "model_traffic.pkl": base_url + "model_traffic.pkl",
    "model_final(point&traffic).pkl": base_url + "model_final(point&traffic).pkl",
    "le_motor.pkl": base_url + "le_motor.pkl",
    "le_mobil.pkl": base_url + "le_mobil.pkl"
}

for local_name, url in model_files.items():
    download_file_from_github(url, local_name)


# ===== PREDIKSI SCORE POINT =====
features = ['Density'] + poi_categories + ['month_ke']
result_df = pd.DataFrame([poi_counts])
result_df['month_ke'] = 3
X_new = result_df[features]

scaler = joblib.load("scaler_point.pkl")
poisson_model = joblib.load("model_point_only.pkl")
X_new_scaled = scaler.transform(X_new)
result_df['pred_walk_in'] = poisson_model.predict(X_new_scaled)
result_df['score_point'] = round(result_df['pred_walk_in'] / 2 * 100, 2).apply(lambda x: min(x, 100))
score_point = result_df['score_point'].values[0]


total_kendaraan = total_motor + total_mobil
X_traffic = pd.DataFrame({'total_traffic': [total_kendaraan], 'lebar_jalan': [lebar_jalan]})
type_motor_enc = model_type_motor.predict(X_traffic)[0]
type_mobil_enc = model_type_mobil.predict(X_traffic)[0]
durasi_motor = model_dur_motor.predict(X_traffic)[0]
durasi_mobil = model_dur_mobil.predict(X_traffic)[0]

X_walkin = pd.DataFrame({
    'total_traffic': [total_kendaraan],
    'pred_type_motor': [type_motor_enc],
    'pred_type_mobil': [type_mobil_enc],
    'pred_dur_motor': [durasi_motor],
    'pred_dur_mobil': [durasi_mobil]
})
predicted_walkin = model_walkin.predict(X_walkin)[0]
score_traffic = (predicted_walkin / 2) * 100

X_final = pd.DataFrame({'score_pos_poi': [score_point], 'score_traffic': [score_traffic]})
final_prediction = model_final.predict(X_final)[0]
score_final = round(final_prediction / 2 * 100, 2)

# ===== OUTPUT =====
st.subheader("\U0001F4CA Hasil Prediksi")
st.write(f"**Kelurahan:** {kelurahan_name}")
st.write(f"**Type Motor:** {le_motor.inverse_transform([type_motor_enc])[0]}")
st.write(f"**Type Mobil:** {le_mobil.inverse_transform([type_mobil_enc])[0]}")
st.write(f"**Durasi Motor:** {durasi_motor:.2f} jam")
st.write(f"**Durasi Mobil:** {durasi_mobil:.2f} jam")
st.write(f"**Score Point:** {score_point:.2f}")
st.write(f"**Score Traffic:** {score_traffic:.2f}")
st.success(f"\U0001F3AF **Score Final Walk-in:** {score_final:.2f}")
