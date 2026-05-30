# EmotiScan — Facial Emotion Detector Web App
### ADBMS Project | Group Members: [Your Name] & Kabeer Kumar

---

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python (Flask)
- **ML Model:** KNN Classifier + OpenCV (Haar Cascade face detection)
- **Database:** MongoDB (NoSQL) via PyMongo

---

## Project Structure

```
emotion_app/
├── app.py              ← Flask backend + all API routes
├── ml_model.py         ← Feature extraction + KNN model logic
├── train_model.py      ← One-time training script
├── requirements.txt    ← Python dependencies
├── model.pkl           ← Saved KNN model (generated after training)
├── templates/
│   ├── login.html      ← Login page
│   ├── register.html   ← Registration page
│   └── dashboard.html  ← Main app (upload, predict, history)
└── static/
    └── uploads/        ← Temporary image storage
```

---

## MongoDB Collections

### `users`
```json
{
  "_id": ObjectId,
  "name": "string",
  "email": "string",
  "password": "bcrypt_hash",
  "created_at": "datetime"
}
```

### `predictions`
```json
{
  "_id": ObjectId,
  "user_id": ObjectId,
  "prediction": "Happy | Sad",
  "confidence": 87.5,
  "happy_pct": 87.5,
  "sad_pct": 12.5,
  "emoji": "😊",
  "image_b64": "base64_string",
  "timestamp": "datetime"
}
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MongoDB
Make sure MongoDB is running locally:
```bash
# Windows
net start MongoDB

# Mac/Linux
sudo systemctl start mongod
```

### 3. Train the Model
Point to your dataset folder (must have Happy/ and Sad/ subfolders):
```bash
python train_model.py --dataset /path/to/your/dataset
```
This creates `model.pkl`.

### 4. Run the App
```bash
python app.py
```

### 5. Open in Browser
```
http://localhost:5000
```

---

## API Routes

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Redirect to login or dashboard |
| GET/POST | `/register` | User registration |
| GET/POST | `/login` | User login |
| GET | `/logout` | Clear session |
| GET | `/dashboard` | Main app page |
| POST | `/predict` | Upload image → get prediction |
| GET | `/history` | Paginated prediction history (JSON) |
| GET | `/stats` | Aggregated stats per user (JSON) |

---

## How It Works

1. User uploads a face image via the web interface
2. Flask receives the image and passes it to `ml_model.py`
3. OpenCV detects the face using Haar Cascade and crops it
4. The face is divided into a 4×4 grid → 32 features (mean + std per cell)
5. KNN model (trained on your dataset) classifies as Happy or Sad
6. Result + confidence scores are saved to MongoDB
7. Dashboard shows result with animated confidence bars + history

---

## Notes for Report
- **NoSQL choice:** MongoDB is used because predictions are document-oriented,
  schema-less (easy to add new emotion categories later), and scales well.
- **Indexes:** `user_id` and `timestamp` are indexed for fast history queries.
- **Aggregation:** MongoDB aggregation pipeline is used for per-user stats.
