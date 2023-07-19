import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Directory paths
cancerous_dir = 'CT-Scan Images/Cancerous raw images-jpg'
non_cancerous_dir = 'CT-Scan Images/Non-Cancerous raw images - jpg'

# Load images and labels
def load_images_and_labels(directories):
    images = []
    labels = []
    for directory in directories:
        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                img = cv2.resize(img, (64, 64))  # Resize image
                images.append(img)
                if 'Non-Cancerous' in directory:
                    labels.append(0)
                else:
                    labels.append(1)
    return np.array(images), np.array(labels)

X, y = load_images_and_labels([cancerous_dir, non_cancerous_dir])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
