from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import cv2
import numpy as np
import os
import base64
from ml_model import load_model, predict_emotion

# ─────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'  # Change this!
bcrypt = Bcrypt(app)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# ─────────────────────────────────────────────
#  MongoDB Connection
# ─────────────────────────────────────────────
client = MongoClient(os.environ.get('MONGO_URI', 'mongodb://localhost:27017/'))
db = client['emotion_detector']

users_col     = db['users']
predictions_col = db['predictions']

# Indexes for faster queries
users_col.create_index('email', unique=True)
predictions_col.create_index('user_id')
predictions_col.create_index('timestamp')

# ─────────────────────────────────────────────
#  Load ML Model
# ─────────────────────────────────────────────

try:
    knn, scaler = load_model('model.pkl')
    print("✅ ML model loaded successfully.")
except FileNotFoundError:
    print("⚠️  model.pkl not found. Train the model first with ml_model.py")
    knn, scaler = None, None

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    doc['_id'] = str(doc['_id'])
    if 'user_id' in doc:
        doc['user_id'] = str(doc['user_id'])
    return doc

# ─────────────────────────────────────────────
#  Auth Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    data = request.get_json()
    name     = data.get('name', '').strip()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not all([name, email, password]):
        return jsonify({'error': 'All fields are required.'}), 400

    if users_col.find_one({'email': email}):
        return jsonify({'error': 'Email already registered.'}), 409

    hashed = bcrypt.generate_password_hash(password).decode('utf-8')
    user = {
        'name': name,
        'email': email,
        'password': hashed,
        'created_at': datetime.now(timezone.utc)
    }
    result = users_col.insert_one(user)
    session['user_id'] = str(result.inserted_id)
    session['user_name'] = name

    return jsonify({'message': 'Account created!', 'redirect': '/dashboard'}), 201


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data = request.get_json()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    user = users_col.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid email or password.'}), 401

    session['user_id'] = str(user['_id'])
    session['user_name'] = user['name']

    return jsonify({'message': 'Logged in!', 'redirect': '/dashboard'}), 200


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ─────────────────────────────────────────────
#  Main App Routes
# ─────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = ObjectId(session['user_id'])
    history = list(predictions_col.find(
        {'user_id': user_id},
        sort=[('timestamp', -1)],
        limit=10
    ))
    history = [serialize_doc(h) for h in history]

    total = predictions_col.count_documents({'user_id': user_id})
    happy_count = predictions_col.count_documents({'user_id': user_id, 'prediction': 'Happy'})
    sad_count   = predictions_col.count_documents({'user_id': user_id, 'prediction': 'Sad'})

    stats = {
        'total': total,
        'happy': happy_count,
        'sad': sad_count
    }

    return render_template('dashboard.html',
                           user_name=session['user_name'],
                           history=history,
                           stats=stats)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if knn is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file. Use PNG, JPG, or JPEG.'}), 400

    # Read image into numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Could not read image.'}), 400

    # Run prediction
    result = predict_emotion(img, knn, scaler)

    if result is None:
        return jsonify({'error': 'No face detected in the image. Please use a clear frontal face photo.'}), 422

    # Encode image as base64 for storing preview
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Save to MongoDB
    doc = {
        'user_id': ObjectId(session['user_id']),
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'happy_pct': result['happy_pct'],
        'sad_pct': result['sad_pct'],
        'emoji': result['emoji'],
        'image_b64': img_b64,
        'timestamp': datetime.now(timezone.utc)
    }
    inserted = predictions_col.insert_one(doc)

    return jsonify({
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'happy_pct': result['happy_pct'],
        'sad_pct': result['sad_pct'],
        'emoji': result['emoji'],
        'record_id': str(inserted.inserted_id)
    }), 200


@app.route('/history')
@login_required
def history():
    user_id = ObjectId(session['user_id'])
    page = int(request.args.get('page', 1))
    per_page = 12
    skip = (page - 1) * per_page

    records = list(predictions_col.find(
        {'user_id': user_id},
        sort=[('timestamp', -1)],
        skip=skip,
        limit=per_page
    ))
    records = [serialize_doc(r) for r in records]
    total = predictions_col.count_documents({'user_id': user_id})

    return jsonify({
        'records': records,
        'total': total,
        'page': page,
        'pages': (total + per_page - 1) // per_page
    })


@app.route('/stats')
@login_required
def stats():
    user_id = ObjectId(session['user_id'])
    pipeline = [
        {'$match': {'user_id': user_id}},
        {'$group': {
            '_id': '$prediction',
            'count': {'$sum': 1},
            'avg_confidence': {'$avg': '$confidence'}
        }}
    ]
    agg = list(predictions_col.aggregate(pipeline))
    return jsonify(agg)


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
