from flask import Flask, request, jsonify, render_template
import cv2
import easyocr
import numpy as np
import re

app = Flask(__name__)

# Inisialisasi model EasyOCR (dilakukan sekali saat server dinyalakan)
print("Memuat model EasyOCR, mohon tunggu sebentar...")
reader = easyocr.Reader(['en'], gpu=False)
print("Model OCR siap!")

@app.route('/')
def home():
    # Merender file HTML Anda yang harus ditaruh di dalam folder 'templates'
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'Tidak ada file gambar yang dikirim'})
    
    file = request.files['image']
    
    # Konversi file yang di-upload ke format numpy array untuk OpenCV
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'success': False, 'message': 'Gagal membaca gambar'})

    # --- TAHAP PREPROCESSING GAMBAR ---
    # Ubah gambar ke Grayscale (hitam-putih)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding untuk mempertajam angka timbangan terhadap background
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    
    # --- EKSEKUSI OCR ---
    results = reader.readtext(thresh)
    
    detected_text = ""
    for (bbox, text, prob) in results:
        if prob > 0.35:  # Ambil teks dengan tingkat keyakinan > 35%
            detected_text += text + " "

    # --- POST-PROCESSING (Pembersihan Teks) ---
    # Hanya ambil karakter angka (0-9) dan titik desimal (.)
    clean_number = re.sub(r'[^0-9.]', '', detected_text)
    
    # Fallback: Jika dengan thresholding kosong, coba baca gambar grayscale asli
    if not clean_number:
        results_raw = reader.readtext(gray)
        raw_text = "".join([t[1] for t in results_raw if t[2] > 0.3])
        clean_number = re.sub(r'[^0-9.]', '', raw_text)

    return jsonify({
        'success': True,
        'weight': clean_number if clean_number else ""
    })

if __name__ == '__main__':
    # Menjalankan server lokal di port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)