import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import requests
from datetime import datetime

ADMIN_BOT_TOKEN = "8101962639:AAHfBs21gifDIQK6BKxBLrMNeU7wMJA1Yj0"
SECURITY_BOT_TOKEN = "7993325231:AAG0rtpJFCgMQ7mVaI_1OKD2rRkiNkMDKiw"
ADMIN_CHAT_ID = "5891007214"
SECURITY_CHAT_ID = "5891007214"

def escape_markdown_v2(text):
    characters_to_escape = r'_*[]()~`>#+-=|{}.!'
    for char in characters_to_escape:
        text = text.replace(char, f'\\{char}')
    return text

def send_to_telegram(bot_token, chat_id, message, image_path=None):
    base_url = f"https://api.telegram.org/bot{bot_token}"
    
    escaped_message = escape_markdown_v2(message)
    
    text_url = f"{base_url}/sendMessage"
    params = {
        "chat_id": chat_id,
        "text": escaped_message,
        "parse_mode": "MarkdownV2"
    }
    response = requests.post(text_url, params=params)
    
    if response.status_code != 200:
        print(f"Gagal mengirim pesan ke chat ID {chat_id}: {response.text}")
        return False
    
    if image_path and os.path.exists(image_path):
        photo_url = f"{base_url}/sendPhoto"
        with open(image_path, 'rb') as img_file:
            files = {'photo': img_file}
            photo_params = {
                "chat_id": chat_id,
                "caption": escape_markdown_v2(f"Gambar kendaraan: {os.path.basename(image_path)}")
            }
            photo_response = requests.post(photo_url, params=photo_params, files=files)
        
        if photo_response.status_code != 200:
            print(f"Gagal mengirim gambar ke chat ID {chat_id}: {photo_response.text}")
            return False
    
    print(f"Berhasil mengirim ke chat ID {chat_id}")
    return True

dataset_path = r"C:\Users\T14s\Documents\python_Kp\archive_vehicle"  
model_path = "vehicle_classifier.keras" 

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset tidak ditemukan di {dataset_path}")
    exit()

if not os.path.exists(model_path):
    print(f"ERROR: Model tidak ditemukan di {model_path}")
    exit()

print(f"Dataset ditemukan di: {dataset_path}")
print(f"Model ditemukan di: {model_path}")

img_size = 224
batch_size = 32

existing_folders = []
for folder in ['test']:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        existing_folders.append(folder)

test_folder_name = 'test' if 'test' in existing_folders else None

if not test_folder_name:
    print("ERROR: Tidak ada folder test ditemukan!")
    exit()

print(f"Menggunakan folder test: {test_folder_name}")

test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading test data generator...")
try:
    test_generator = test_datagen.flow_from_directory(
        os.path.join(dataset_path, test_folder_name),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    class_labels = list(test_generator.class_indices.keys())
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

print("Test data generator berhasil dibuat")
print("Kelas yang ditemukan:", test_generator.class_indices)
print("Jumlah kelas:", test_generator.num_classes)

print("Loading model...")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Model berhasil diload")
model.summary()

def capture_image_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Tidak dapat mengakses webcam!")
        return None
    
    print("Kamera aktif. Tekan 's' untuk mengambil gambar, 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Gagal menangkap gambar dari webcam!")
            cap.release()
            return None
        
        cv2.imshow('Webcam - Tekan "s" untuk simpan, "q" untuk keluar', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            temp_dir = "temp_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(temp_dir, f"webcam_capture_{timestamp}.jpg")
            
            cv2.imwrite(image_path, frame)
            print(f"Gambar disimpan di: {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return image_path
        elif key == ord('q'):
            print("Pengambilan gambar dibatalkan.")
            cap.release()
            cv2.destroyAllWindows()
            return None

def predict_single_image(image_path, model, img_size=224):
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    results = {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'all_predictions': {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    }
    
    return results

def tampilkan_detail_kendaraan(jenis_kendaraan):
    detail_kendaraan = {
        'bus': {
            'nama': 'Bus',
            'deskripsi': 'Kendaraan angkutan umum berkapasitas besar',
            'karakteristik': 'Ukuran besar, bentuk kotak, banyak jendela, warna biasanya kuning/biru/merah'
        },
        'mobil': {
            'nama': 'Mobil Penumpang',
            'deskripsi': 'Kendaraan pribadi untuk transportasi personal',
            'karakteristik': 'Ukuran sedang, 4-5 pintu, berbagai bentuk dan warna'
        },
        'motor': {
            'nama': 'Sepeda Motor',
            'deskripsi': 'Kendaraan roda dua untuk transportasi personal',
            'karakteristik': 'Ukuran kecil, 2 roda, setang, berbagai jenis (sport/matic/bebek)'
        },
        'truck': {
            'nama': 'Truk',
            'deskripsi': 'Kendaraan angkutan barang berat',
            'karakteristik': 'Ukuran sangat besar, bak terbuka/tertutup, roda banyak, warna biasanya gelap'
        }
    }
    
    if jenis_kendaraan.lower() in detail_kendaraan:
        info = detail_kendaraan[jenis_kendaraan.lower()]
        print(f"DETAIL JENIS KENDARAAN:")
        print(f"   Nama: {info['nama']}")
        print(f"   Deskripsi: {info['deskripsi']}")
        print(f"   Karakteristik: {info['karakteristik']}")
    else:
        print(f"Jenis kendaraan '{jenis_kendaraan}' tidak memiliki info detail")

def display_prediction(image_path, results):
    img = Image.open(image_path)
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'GAMBAR: {os.path.basename(image_path)}', fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = list(results['all_predictions'].keys())
    probabilities = list(results['all_predictions'].values())
    
    colors = ['lightblue' if cls != results['predicted_class'] else 'steelblue' 
              for cls in classes]
    
    bars = plt.barh(classes, probabilities, color=colors)
    plt.xlabel('PROBABILITAS', fontweight='bold')
    plt.title('HASIL PREDIKSI MODEL', fontweight='bold', pad=20)
    plt.xlim(0, 1)
    
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*70}")
    print("HASIL KLASIFIKASI KENDARAAN")
    print(f"{'='*70}")
    print(f"   File: {os.path.basename(image_path)}")
    print(f"   Jenis Kendaraan: {results['predicted_class'].upper()}")
    print(f"   Tingkat Keyakinan: {results['confidence']*100:.2f}%")
    
    tampilkan_detail_kendaraan(results['predicted_class'])
    
    print(f"\nDETAIL PROBABILITAS SEMUA KELAS:")
    for cls, prob in sorted(results['all_predictions'].items(), key=lambda x: x[1], reverse=True):
        bintang = "*" if cls == results['predicted_class'] else " "
        print(f"   {bintang} {cls}: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "="*70)
print("SISTEM KLASIFIKASI JENIS KENDARAAN")
print("="*70)

print("\nSiapkan kamera untuk menangkap gambar kendaraan...")
print("   (Tekan 's' untuk mengambil gambar, 'q' untuk keluar)")

image_path = capture_image_from_webcam()

if not image_path:
    print("Tidak ada gambar yang diambil!")
    exit()

if not os.path.exists(image_path):
    print("File tidak ditemukan!")
    exit()

try:
    print(f"Memproses gambar: {os.path.basename(image_path)}")
    
    results = predict_single_image(image_path, model, img_size)
    
    display_prediction(image_path, results)
    
    predicted_class = results['predicted_class']
    confidence = results['confidence'] * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    message = f"**HASIL KLASIFIKASI KENDARAAN**\n" \
              f"Waktu: {timestamp}\n" \
              f"File: {os.path.basename(image_path)}\n" \
              f"Jenis: {predicted_class.upper()}\n" \
              f"Keyakinan: {confidence:.2f}%\n\n" \
              f"Detail Probabilitas:\n"
    
    for cls, prob in sorted(results['all_predictions'].items(), key=lambda x: x[1], reverse=True):
        star = "*" if cls == predicted_class else ""
        message += f"{star} {cls}: {prob*100:.2f}%\n"
    
    print("Mengirim hasil ke Telegram...")
    
    send_to_telegram(SECURITY_BOT_TOKEN, SECURITY_CHAT_ID, message, image_path)
    
    if predicted_class.lower() == 'truck':
        print("Kendaraan terdeteksi sebagai TRUCK, mengirim ke admin juga...")
        send_to_telegram(ADMIN_BOT_TOKEN, ADMIN_CHAT_ID, message, image_path)
    else:
        print(f"Kendaraan terdeteksi sebagai {predicted_class.upper()}, hanya dikirim ke security.")

except Exception as e:
    print(f"Error selama prediksi: {e}")
    print("Pastikan gambar yang diambil valid.")

print("\n" + "="*70)
print("KLASIFIKASI SELESAI! TERIMA KASIH TELAH MENGGUNAKAN SISTEM INI")
print("="*70)