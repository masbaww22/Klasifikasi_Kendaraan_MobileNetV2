import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import requests
from datetime import datetime
import time 

latency_log = []

ADMIN_BOT_TOKEN = "8101962639:AAHfBs21gifDIQK6BKxBLrMNeU7wMJA1Yj0"
SECURITY_BOT_TOKEN = "7993325231:AAG0rtpJFCgMQ7mVaI_1OKD2rRkiNkMDKiw"
ADMIN_CHAT_ID = "5891007214"
SECURITY_CHAT_ID = "5891007214"

def send_to_telegram(bot_token, chat_id, message, image_path=None):
    base_url = f"https://api.telegram.org/bot{bot_token}"
    print(f"DEBUG: Mengirim ke chat {chat_id}")
    print(f"Panjang pesan: {len(message)}")
    print(f"Pesan preview: {message[:200]}...")

    total_start = time.time()

    text_url = f"{base_url}/sendMessage"
    params = {"chat_id": chat_id, "text": message}

    start_text = time.time()
    response = requests.post(text_url, params=params)
    end_text = time.time()
    print(f"⏱️ Durasi kirim pesan teks: {end_text - start_text:.3f} detik")

    if response.status_code != 200:
        error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
        print(f"Gagal mengirim pesan ke chat ID {chat_id}: {error_detail}")
        return False

    print(f"Pesan berhasil dikirim ke chat ID {chat_id}")

    if image_path and os.path.exists(image_path):
        photo_url = f"{base_url}/sendPhoto"
        with open(image_path, 'rb') as img_file:
            files = {'photo': img_file}
            photo_params = {"chat_id": chat_id, "caption": f"Gambar kendaraan: {os.path.basename(image_path)}"}

            start_photo = time.time()
            photo_response = requests.post(photo_url, params=photo_params, files=files)
            end_photo = time.time()
            print(f"⏱️ Durasi kirim gambar: {end_photo - start_photo:.3f} detik")

        if photo_response.status_code != 200:
            print(f"Gagal mengirim gambar ke chat ID {chat_id}: {photo_response.text}")
            return False

        print(f"Gambar berhasil dikirim ke chat ID {chat_id}")

    total_end = time.time()
    total_duration = total_end - total_start
    print(f"✅ Total waktu komunikasi ke Telegram: {total_duration:.3f} detik\n")

    latency_log.append({
        "chat_id": chat_id,
        "file": os.path.basename(image_path) if image_path else "tanpa_gambar",
        "latency": total_duration
    })

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

def pilih_gambar_dari_explorer():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar Kendaraan untuk Diklasifikasi",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    return file_path

def pilih_gambar_per_kelas(base_folder, class_labels):
    root = tk.Tk()
    root.withdraw()
    selected_files = {}
    print("\nPilih satu gambar dari setiap kelas:")
    for cls in class_labels:
        print(f"   Pilih gambar untuk kelas '{cls}'...")
        initial_dir = os.path.join(base_folder, cls)
        file_path = filedialog.askopenfilename(
            title=f"Pilih gambar untuk kelas {cls}",
            initialdir=initial_dir,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            selected_files[cls] = file_path
            print(f"✅ Dipilih: {os.path.basename(file_path)}")
        else:
            print(f"⚠️ Tidak ada gambar dipilih untuk {cls}, dilewati.")
    return selected_files


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

def plot_latency_graph(latency_log):
    if not latency_log:
        print("Tidak ada data latency untuk ditampilkan.")
        return

    plt.figure(figsize=(10, 6))
    files = [item['file'] for item in latency_log]
    latencies = [item['latency'] for item in latency_log]
    plt.barh(files, latencies)
    plt.xlabel("Waktu Komunikasi (detik)")
    plt.ylabel("Nama File")
    plt.title("Grafik Latency Pengiriman Notifikasi ke Telegram")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    for i, val in enumerate(latencies):
        plt.text(val + 0.01, i, f"{val:.2f}s", va='center')
    plt.tight_layout()
    plt.show()

print("\n" + "="*70)
print("SISTEM KLASIFIKASI JENIS KENDARAAN")
print("="*70)

print("\nSilakan pilih satu gambar dari setiap kelas kendaraan...")
selected_images = pilih_gambar_per_kelas(os.path.join(dataset_path, test_folder_name), class_labels)

for cls, image_path in selected_images.items():
    if not os.path.exists(image_path):
        print(f"File {image_path} tidak ditemukan! Dilewati.")
        continue

    try:
        print(f"\nMemproses gambar: {os.path.basename(image_path)}")
        results = predict_single_image(image_path, model, img_size)
        predicted_class = results['predicted_class']
        confidence = results['confidence'] * 100
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
        message = f"HASIL KLASIFIKASI KENDARAAN\n" \
                  f"Waktu: {timestamp}\n" \
                  f"File: {os.path.basename(image_path)}\n" \
                  f"Jenis: {predicted_class.upper()}\n" \
                  f"Keyakinan: {confidence:.2f}%\n\n" \
                  f"Detail Probabilitas:\n"
        for c, prob in sorted(results['all_predictions'].items(), key=lambda x: x[1], reverse=True):
            star = "* " if c == predicted_class else "  "
            message += f"{star}{c}: {prob*100:.2f}%\n"

        print("Mengirim hasil ke Telegram...")
        send_to_telegram(SECURITY_BOT_TOKEN, SECURITY_CHAT_ID, message, image_path)

        if predicted_class.lower() == 'truck':
            print("Kendaraan terdeteksi sebagai TRUCK, mengirim ke admin juga...")
            send_to_telegram(ADMIN_BOT_TOKEN, ADMIN_CHAT_ID, message, image_path)
        else:
            print(f"Kendaraan terdeteksi sebagai {predicted_class.upper()}, hanya dikirim ke security.")
    except Exception as e:
        print(f"Error selama prediksi {image_path}: {e}")

print("\nMenampilkan grafik latency notifikasi Telegram...")
plot_latency_graph(latency_log)

print("\n" + "="*70)
print("KLASIFIKASI SELESAI! TERIMA KASIH TELAH MENGGUNAKAN SISTEM INI")
print("="*70)
