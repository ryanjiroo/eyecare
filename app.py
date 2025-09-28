import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__, template_folder='.')

# Muat model Keras
try:
    # Model Anda memiliki arsitektur yang sudah ditentukan di file model
    model = tf.keras.models.load_model('eye_model_resnet50.keras')
    class_names = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Uveitis', 'normal']
    img_size = (224, 224)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None

# Fungsi untuk preprocessing gambar
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    return tf.keras.applications.resnet50.preprocess_input(np.expand_dims(img_array, axis=0))

@app.route('/')
def home():
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
        # Gunakan jalur sementara yang aman di Vercel
        temp_dir = '/tmp'
        os.makedirs(temp_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        try:
            # Preprocessing gambar
            processed_image = preprocess_image(filepath)
            
            # Lakukan prediksi
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Hapus file setelah prediksi
            os.remove(filepath)
    return jsonify({'error': 'Terjadi kesalahan saat mengunggah file'}), 500

if __name__ == '__main__':
    # Vercel tidak menjalankan kode ini
    # Ini hanya untuk pengujian lokal
    app.run(debug=True)