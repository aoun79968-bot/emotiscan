import cv2
import numpy as np
import os
import glob
import pickle
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────

def grid_features_with_face(image_path=None, img_array=None):
    """
    Extract 32 grid-based features from a face in an image.
    Accepts either a file path or a numpy array (for uploaded images).
    Returns None if no face is detected.
    """
    if image_path is not None:
        img_color = cv2.imread(image_path)
    elif img_array is not None:
        img_color = img_array
    else:
        return None

    if img_color is None:
        return None

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    img_gray = img_gray[y:y+h, x:x+w]

    img = cv2.resize(img_gray, (128, 128))
    features = []
    grid_size = 4
    cell_size = 128 // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            cell = img[row*cell_size:(row+1)*cell_size, col*cell_size:(col+1)*cell_size]
            features.append(np.mean(cell))
            features.append(np.std(cell))

    return features


# ─────────────────────────────────────────────
#  Model Training
# ─────────────────────────────────────────────

def train_model(dataset_path, model_save_path='model.pkl'):
    """
    Train KNN model on dataset and save to disk.
    Dataset should have subfolders: Sad/, Happy/
    Returns (accuracy, confusion_matrix, model, scaler)
    """
    features = []
    labels = []
    emotion_list = ['Sad', 'Happy']

    for label_idx, emotion in enumerate(emotion_list):
        folder = os.path.join(dataset_path, emotion)
        image_files = glob.glob(os.path.join(folder, '*.*'))

        for image in image_files:
            feat = grid_features_with_face(image_path=image)
            if feat is not None:
                features.append(feat)
                labels.append(label_idx)

    features = np.array(features)
    labels = np.array(labels)

    print(f"Total images processed: {len(features)}")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)

    y_pred = knn.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Save model + scaler
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': knn, 'scaler': scaler}, f)

    print(f"Model saved to {model_save_path}")
    print(f"Accuracy: {accuracy*100:.2f}%")

    return accuracy, cm, knn, scaler


# ─────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────

def load_model(model_path='model.pkl'):
    """Load saved KNN model and scaler from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train the model first.")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data['model'], data['scaler']


# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────

def predict_emotion(img_array, knn, scaler):
    """
    Predict emotion from a numpy image array.
    Returns dict with prediction, confidence, and probabilities.
    Returns None if no face detected.
    """
    features = grid_features_with_face(img_array=img_array)

    if features is None:
        return None

    features_np = np.array([features])
    features_scaled = scaler.transform(features_np)

    proba = knn.predict_proba(features_scaled)[0]
    sad_conf = proba[0] * 100
    happy_conf = proba[1] * 100

    if happy_conf >= sad_conf:
        prediction = "Happy"
        confidence = happy_conf
        emoji = "😊"
    else:
        prediction = "Sad"
        confidence = sad_conf
        emoji = "😢"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "happy_pct": round(happy_conf, 2),
        "sad_pct": round(sad_conf, 2),
        "emoji": emoji
    }
