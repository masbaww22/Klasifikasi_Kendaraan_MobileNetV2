import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np

dataset_path = r"C:\Users\T14s\Documents\python_Kp\archive_vehicle" 

if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Please ensure the dataset is placed in the specified directory.")
    exit()
else:
    print(f"Dataset found at: {dataset_path}")

def explore_dataset_structure(base_path):
    print("Dataset structure:")
    print("=" * 50)
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"{item}/")
            for subitem in os.listdir(item_path)[:5]:
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    print(f"   ├── {subitem}/")
                    images = [f for f in os.listdir(subitem_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"   │    └── {len(images)} images")
                else:
                    print(f"   ├── {subitem}")
        else:
            print(f"{item}")

explore_dataset_structure(dataset_path)

def show_random_images_from_subfolder(base_path, subset, samples_per_class=2):
    path = os.path.join(base_path, subset)
    if not os.path.exists(path):
        print(f"Folder '{subset}' not found at {base_path}")
        return
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not classes:
        print(f"No classes found in {path}")
        return
    for cls in classes:
        cls_path = os.path.join(path, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            print(f"No images found in {cls_path}")
            continue
        sampled_images = random.sample(images, min(samples_per_class, len(images)))
        plt.figure(figsize=(10, 4))
        plt.suptitle(f"{subset.capitalize()} - {cls}", fontsize=16)
        for i, img_name in enumerate(sampled_images):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path)
                plt.subplot(1, samples_per_class, i+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"{img_name}\n{img.size[0]}x{img.size[1]}", fontsize=8)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        plt.tight_layout()
        plt.show()

print("Displaying sample images:")
existing_folders = []
for folder in ['train', 'val', 'test', 'Dataset']:
    folder_path = os.path.join(dataset_path, folder)
    if os.path.exists(folder_path):
        existing_folders.append(folder)
        show_random_images_from_subfolder(dataset_path, folder)

print(f"Folders found: {existing_folders}")

train_folder_name = 'train' if 'train' in existing_folders else 'Dataset' if 'Dataset' in existing_folders else None
val_folder_name = 'val' if 'val' in existing_folders else None
test_folder_name = 'test' if 'test' in existing_folders else None

print(f"Using folders:")
print(f"   Train: {train_folder_name}")
print(f"   Validation: {val_folder_name}")
print(f"   Test: {test_folder_name}")

if not train_folder_name:
    print("ERROR: No training folder found!")
    exit()

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("Loading data generators...")
try:
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, train_folder_name),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
except Exception as e:
    print(f"Error loading training data: {e}")
    exit()

val_generator = None
if val_folder_name:
    try:
        val_generator = val_datagen.flow_from_directory(
            os.path.join(dataset_path, val_folder_name),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
    except Exception as e:
        print(f"Error loading validation data: {e}")

test_generator = None
if test_folder_name:
    try:
        test_generator = test_datagen.flow_from_directory(
            os.path.join(dataset_path, test_folder_name),
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical'
        )
    except Exception as e:
        print(f"Error loading test data: {e}")

print("Data generators created successfully")
print("Classes found:", train_generator.class_indices)
print("Number of classes:", train_generator.num_classes)

print("Creating model...")
try:
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    print(f"Error creating model: {e}")
    exit()

early_stopping = EarlyStopping(
    monitor='val_loss' if val_generator else 'loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

callbacks = [early_stopping]

if val_generator:
    checkpoint = ModelCheckpoint(
        "best_vehicle_classifier.keras", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)

print("Model created and compiled successfully")
model.summary()

print("Starting training...")
try:
    if val_generator:
        print("Training with validation data")
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("Training without validation data")
        history = model.fit(
            train_generator,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )
except Exception as e:
    print(f"Error during training: {e}")
    exit()

try:
    model.save("vehicle_classifier.keras")
    print("Model saved at: vehicle_classifier.keras")
except Exception as e:
    print(f"Error saving model: {e}")

if test_generator:
    print("Evaluating on test data:")
    try:
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
    except Exception as e:
        print(f"Error evaluating test data: {e}")
else:
    print("No test data available for evaluation")

if history:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    if val_generator and 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    if val_generator and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("Training completed!")