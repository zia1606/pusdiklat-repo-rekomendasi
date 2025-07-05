import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/count-by-year', methods=['POST'])
def count_by_year():
    try:
        data = request.json
        
        # Validasi input
        if not data or 'collections' not in data or 'reference_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Format data tidak valid'
            }), 400

        # Cari tahun terbit referensi
        reference_year = None
        for collection in data['collections']:
            if collection['id'] == data['reference_id']:
                reference_year = collection['tahun_terbit']
                break
        
        if not reference_year:
            return jsonify({
                'status': 'error',
                'message': 'Koleksi referensi tidak ditemukan'
            }), 404

        # Hitung koleksi dengan tahun sama
        count = sum(1 for c in data['collections'] 
                   if c['tahun_terbit'] == reference_year and 
                   c['id'] != data['reference_id'])

        return jsonify({
            'status': 'success',
            'reference_id': data['reference_id'],
            'reference_year': reference_year,
            'same_year_count': count,
            'collections': [
                {
                    'id': c['id'],
                    'judul': c['judul'],
                    'tahun_terbit': c['tahun_terbit']
                } 
                for c in data['collections'] 
                if c['tahun_terbit'] == reference_year
            ]
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("WEB_PORT", 5000))
    app.run(host='0.0.0.0', port=port)