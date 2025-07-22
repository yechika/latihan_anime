import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity # Import yang dibutuhkan

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- PEMUATAN MODEL (DILAKUKAN SEKALI SAAT STARTUP) ---
print("Memuat data dan matriks fitur...")
try:
    # Ganti nama file pkl jika Anda menamainya berbeda
    data = pickle.load(open('anime_recommender.pkl', 'rb')) 
    df_clean = data['anime_list']
    feature_matrix = data['feature_matrix'] # Sesuaikan dengan kunci yang benar
    
    # Buat Series untuk memetakan judul anime ke indeksnya agar pencarian cepat
    indices = pd.Series(df_clean.index, index=df_clean['Name']).drop_duplicates()
    print("Model berhasil dimuat.")

except FileNotFoundError:
    print("Error: File 'anime_recommender.pkl' tidak ditemukan.")
    exit()
except KeyError:
    print("Error: Pastikan file .pkl berisi 'anime_list' dan 'similarity_matrix'.")
    exit()


# --- API ENDPOINT UNTUK REKOMENDASI ---
@app.route('/recommend', methods=['GET'])
def recommend():
    # Ambil nama anime dari parameter URL
    anime_title = request.args.get('anime')

    if not anime_title:
        return jsonify({"error": "Parameter 'anime' tidak ditemukan."}), 400

    # --- LOGIKA PENCARIAN FLEKSIBEL ---
    # Cek apakah judul ada persis di dalam data
    if anime_title not in indices:
        # Jika tidak, coba cari yang mengandung kata kunci tersebut
        possible_matches = df_clean[df_clean['Name'].str.contains(anime_title, case=False)]
        if len(possible_matches) == 0:
            return jsonify({"error": f"Anime dengan judul '{anime_title}' tidak ditemukan."}), 404
        # Ambil hasil pertama yang paling cocok
        anime_title = possible_matches.iloc[0]['Name']

    try:
        # Dapatkan indeks dari anime yang cocok
        idx = indices[anime_title]
        
        # Ambil vektor fitur untuk anime target
        target_vector = feature_matrix[idx:idx+1]
        
        # --- HITUNG COSINE SIMILARITY SECARA ON-THE-FLY ---
        sim_scores = cosine_similarity(target_vector, feature_matrix)

        # Proses sisa logika seperti biasa
        sim_scores = list(enumerate(sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        anime_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Format hasil rekomendasi
        recommendations = []
        for i in range(len(anime_indices)):
            name = df_clean['Name'].iloc[anime_indices[i]]
            score = similarity_scores[i]
            recommendations.append(f"{name} ({score*100:.2f}%)")

        return jsonify({
            "source_anime": anime_title,
            "recommendations": recommendations
        })

    except Exception as e:
        # Tangani error tak terduga
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500


# --- JALANKAN APLIKASI ---
if __name__ == '__main__':
    app.run(port=5000, debug=False) 