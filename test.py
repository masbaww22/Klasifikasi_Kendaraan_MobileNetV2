import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from PIL import Image
import tkinter as tk
from tkinter import filedialog

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

print("Evaluasi pada test data:")
try:
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
except Exception as e:
    print(f"Error selama evaluasi: {e}")

print("Melakukan prediksi...")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

def hitung_dan_plot_metrik(y_true, y_pred, label_kelas):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(label_kelas)))

    print("\nMETRIK KLASIFIKASI PER KELAS:")
    print("=" * 70)
    for i, label in enumerate(label_kelas):
        print(f"   {label:<15} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1-Score: {f1[i]:.4f}")
    print("=" * 70)
    print(f"   Rata-rata Macro Precision : {np.mean(precision):.4f}")
    print(f"   Rata-rata Macro Recall    : {np.mean(recall):.4f}")
    print(f"   Rata-rata Macro F1-Score  : {np.mean(f1):.4f}")
    print("=" * 70)

    x = np.arange(len(label_kelas))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1, width, label='F1-Score', color='salmon')

    plt.xticks(x, label_kelas, rotation=20, ha='right')
    plt.ylim(0, 1.1)
    plt.xlabel("Kelas")
    plt.ylabel("Skor")
    plt.title("Perbandingan Precision, Recall, dan F1-Score per Kelas")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
hitung_dan_plot_metrik(true_classes, predicted_classes, class_labels)

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

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

print("\nSilakan pilih gambar kendaraan dari jendela Explorer...")
print("   (Jendela mungkin muncul di belakang, cek taskbar jika tidak kelihatan)")

image_path = pilih_gambar_dari_explorer()

if not image_path:
    print("Tidak ada gambar yang dipilih!")
    exit()

if not os.path.exists(image_path):
    print("File tidak ditemukan!")
    exit()

try:
    print(f"Memproses gambar: {os.path.basename(image_path)}")
    results = predict_single_image(image_path, model, img_size)
    display_prediction(image_path, results)
except Exception as e:
    print(f"Error selama prediksi: {e}")
    print("Pastikan file yang dipilih adalah gambar yang valid.")

print("\n" + "="*70)
print("KLASIFIKASI SELESAI! TERIMA KASIH TELAH MENGGUNAKAN SISTEM INI")
print("="*70)
