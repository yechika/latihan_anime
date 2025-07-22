import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# --- PEMUATAN MODEL ---
# Path ke file .pkl relatif dari lokasi skrip ini
pkl_path = os.path.join(os.path.dirname(__file__), '..', 'anime_recommender.pkl')

try:
    print("Memuat data dan matriks fitur...")
    data = pickle.load(open(pkl_path, 'rb'))
    df_clean = data['anime_list']
    feature_matrix = data['feature_matrix']
    indices = pd.Series(df_clean.index, index=df_clean['Name']).drop_duplicates()
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")


@app.route('/recommend', methods=['GET'])
def recommend():
    # Logika di dalam fungsi ini tetap sama seperti sebelumnya
    anime_title = request.args.get('anime')
    if not anime_title:
        return jsonify({"error": "Parameter 'anime' tidak ditemukan."}), 400

    if anime_title not in indices:
        possible_matches = df_clean[df_clean['Name'].str.contains(anime_title, case=False)]
        if len(possible_matches) == 0:
            return jsonify({"error": f"Anime dengan judul '{anime_title}' tidak ditemukan."}), 404
        anime_title = possible_matches.iloc[0]['Name']

    try:
        idx = indices[anime_title]
        target_vector = feature_matrix[idx:idx+1]
        sim_scores = cosine_similarity(target_vector, feature_matrix)

        sim_scores = list(enumerate(sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        anime_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
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
        return jsonify({"error": f"Terjadi kesalahan internal: {str(e)}"}), 500

@app.route('/')
def home():
    return "API Recommender Berjalan di Vercel!"

# Anda TIDAK memerlukan baris "if __name__ == '__main__':" untuk Vercel
