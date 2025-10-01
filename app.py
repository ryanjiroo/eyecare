import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# NOTE: Di Vercel, kita tidak menggunakan UPLOAD_FOLDER karena tidak disarankan menyimpan file ke disk.

# Muat model Keras
MODEL_PATH = 'eye_model_resnet50.keras'
try:
    # Model Anda memiliki arsitektur yang sudah ditentukan di file model
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Uveitis', 'normal']
    img_size = (224, 224)
    print("Model berhasil dimuat.")
except Exception as e:
    # Ini akan ditampilkan di log Vercel jika model gagal dimuat
    print(f"Error saat memuat model dari {MODEL_PATH}: {e}")
    model = None

# Fungsi untuk preprocessing gambar (Menggunakan bytes/memory)
def preprocess_image(image_bytes):
    # Memuat gambar dari bytes menggunakan PIL
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    # Ubah skala piksel menjadi -1 hingga 1, sesuai yang dibutuhkan oleh ResNet50
    return tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))

@app.route('/')
def home():
    # Menghubungkan ke index.html untuk penggunaan browser
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dimuat.'}), 500
    
    # Pastikan file diunggah
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
    
    file = request.files['file']
    
    # Jika tidak ada file yang dipilih
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    
    if file:
        try:
            # Ambil konten file sebagai bytes
            image_bytes = file.read()
            
            # Preprocessing gambar langsung dari memori
            processed_image = preprocess_image(image_bytes)
            
            # Lakukan prediksi
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            # Terapkan threshold 30% untuk mendeteksi yang "bukan mata"
            threshold = 0.30
            if confidence < threshold:
                predicted_class = 'bukan mata'
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat memproses file: {e}'}), 500

    return jsonify({'error': 'Terjadi kesalahan yang tidak terduga'}), 500

# NOTE: Di Vercel, baris ini TIDAK DIPERLUKAN (atau biarkan saja, Vercel akan mengabaikannya)
# if __name__ == '__main__':
#     app.run(debug=True)

# Tambahkan struktur entry point untuk Vercel (Opsional jika menggunakan vercel.json)
# from vercel_app import app as application # jika menggunakan vercel_app.py
# def handler(event, context):
#     return application(event, context)
