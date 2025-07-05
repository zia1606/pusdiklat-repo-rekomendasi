import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import warnings

app = Flask(__name__)
CORS(app)
warnings.filterwarnings('ignore')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        
        # Validasi input
        if not data or 'collections' not in data or 'reference_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Format data tidak valid'
            }), 400

        # Cari index referensi
        reference_idx = None
        collections = data['collections']
        for i, col in enumerate(collections):
            if col['id'] == data['reference_id']:
                reference_idx = i
                break
        
        if reference_idx is None:
            return jsonify({
                'status': 'error',
                'message': 'Referensi tidak ditemukan'
            }), 404

        # Proses Content-Based Filtering
        df = pd.DataFrame(collections)
        
        # Pastikan ada preprocessing
        if 'preprocessing' not in df.columns:
            return jsonify({
                'status': 'error',
                'message': 'Data preprocessing tidak tersedia'
            }), 400

        # Hitung TF-IDF dengan parameter yang lebih longgar
        tfidf = TfidfVectorizer(
            min_df=1,           # Term muncul minimal 1 dokumen
            max_df=0.9,         # Term muncul di maksimal 90% dokumen
            ngram_range=(1, 3), # Menerima 1-3 kata
            stop_words=None     # Tidak menghapus stop words
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform(df['preprocessing'].fillna(''))
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': 'Tidak cukup data untuk analisis'
            }), 400

        # Hitung cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[reference_idx], tfidf_matrix)
        
        # Dapatkan semua rekomendasi (kecuali referensi itu sendiri)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter untuk mendapatkan minimal similarity > 0
        recommendations = []
        for i, score in sim_scores[1:]:  # Skip referensi (index 0)
            if score > 0:  # Hanya ambil yang memiliki similarity > 0
                collection = df.iloc[i]
                recommendations.append({
                    'id': int(collection['id']),
                    'judul': collection['judul'],
                    'penulis': collection.get('penulis', ''),
                    'kategori': collection.get('kategori', ''),
                    'jenis_dokumen': collection.get('jenis_dokumen', ''),
                    'tahun_terbit': collection.get('tahun_terbit', ''),
                    'similarity_score': float(score)
                })

        # Ambil top N rekomendasi (atau semua jika kurang dari top_n)
        top_n = min(int(data.get('top_n', 5)), len(recommendations))
        top_recommendations = recommendations[:top_n]

        if not top_recommendations:
            return jsonify({
                'status': 'error',
                'message': 'Tidak ditemukan rekomendasi yang sesuai'
            }), 404

        return jsonify({
            'status': 'success',
            'reference_id': data['reference_id'],
            'recommendations': top_recommendations
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("WEB_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)