import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the cancerous and non-cancerous images
cancerous_images = np.load('ct_scan_images/cancerous_images.npy')
non_cancerous_images = np.load('ct_scan_images/non_cancerous_images.npy')

# Combine the cancerous and non-cancerous images
images = np.concatenate((cancerous_images, non_cancerous_images))

# Preprocess the images
images = images.astype('float32')
images /= 255.

# Label the images
labels = np.concatenate((np.ones(len(cancerous_images)), np.zeros(len(non_cancerous_images))))

# Split the images into training and testing sets
(x_train, y_train), (x_test, y_test) = keras.utils.train_test_split(images, labels, test_size=0.2)

# Train a CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('brain_tumor_detection_model.h5')