import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Data preprocessing
data = []
labels = []
classes = 5
cur_path = os.getcwd()

for i in range(classes):
    path = os.path.join(cur_path, 'Dataset', str(i))
    images = os.listdir(path)
    print("Image processing:Class=", (i))
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.convert("RGB")  # Convert to 3-channel RGB
            image = image.resize((120, 120))  # Resize to 120x120
            image = np.array(image, dtype=np.uint8)  # Ensure consistent data type
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)

data = np.array(data)
labels = np.array(labels)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Convert labels to categorical
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Model architecture
model = Sequential()

# Block 1
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:], padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Block 2
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Block 3
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

# Block 4
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))

# Learning rate scheduling
initial_learning_rate = 0.001
epochs = 20
decay_rate = 1e-6
steps_per_epoch = len(X_train) / 128
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=epochs*steps_per_epoch,
    decay_rate=decay_rate,
    staircase=True
)
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    steps_per_epoch=len(X_train) / 128,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# Save the model
model.save("rice_disease_model256.keras")

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
