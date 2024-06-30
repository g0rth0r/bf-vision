import os
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths
data_dir = './frames'
train_data_dir = './train_frames'
validation_data_dir = './val_frames'

# Create directories if they don't exist
os.makedirs(train_data_dir, exist_ok=True)
os.makedirs(validation_data_dir, exist_ok=True)

# Image dimensions
img_width, img_height = 244, 244
max_dim = 570

def random_crop(img, crop_size=(img_width, img_height)):
    crop_height, crop_width = crop_size
    max_x = img.shape[1] - crop_width
    max_y = img.shape[0] - crop_height
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)
    cropped_img = img[y:y + crop_height, x:x + crop_width, :]
    return cropped_img

def preprocess_input(img):
    img = img / 255.0
    img = random_crop(img)
    return img

# Custom ImageDataGenerator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

# Load image paths and labels
image_paths = []
labels = []
class_indices = {}
class_counter = 0

for game in os.listdir(data_dir):
    game_path = os.path.join(data_dir, game)
    if os.path.isdir(game_path):
        for frame in os.listdir(game_path):
            if frame.endswith('.jpg'):
                image_paths.append(os.path.join(game_path, frame))
                if game not in class_indices:
                    class_indices[game] = class_counter
                    class_counter += 1
                labels.append(class_indices[game])

# Convert labels to one-hot encoding
num_classes = len(class_indices)
labels = tf.keras.utils.to_categorical(labels, num_classes)

# Split data into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels)

# Create directories for each class in train and validation directories
for game in class_indices.keys():
    os.makedirs(os.path.join(train_data_dir, game), exist_ok=True)
    os.makedirs(os.path.join(validation_data_dir, game), exist_ok=True)

# Move training files
for train_path in train_paths:
    game_name = train_path.split(os.sep)[-2]
    shutil.copy(train_path, os.path.join(train_data_dir, game_name))

# Move validation files
for val_path in val_paths:
    game_name = val_path.split(os.sep)[-2]
    shutil.copy(val_path, os.path.join(validation_data_dir, game_name))

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(max_dim, max_dim),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(max_dim, max_dim),
    batch_size=32,
    class_mode='categorical'
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
epochs = 25
model.fit(
    train_generator,
    steps_per_epoch=len(train_paths) // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(val_paths) // 32
)

# Save the model
model.save('game_recognition_model.h5')
