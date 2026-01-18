import base64
import numpy as np
import io
import os
import shutil
import threading
from flask import Flask, render_template, request, jsonify

#----------------this is for dashboard-------------
from flask import Flask, render_template, request
from pymongo import MongoClient
import json
from classifier import classify_patient
from datetime import datetime
from staff_calculator import calculate_staff_needed
#----------------------------------------------------

# AI & Image Libraries
from PIL import Image
from bing_image_downloader import downloader
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

app = Flask(__name__)

# ---------- MongoDB Atlas connection ----------
MONGO_URI = "mongodb+srv://er_user:erregistration26@er-registration.g3veiau.mongodb.net/?appName=ER-registration"
client = MongoClient(MONGO_URI)
db = client.er_dashboard
patients = db.patients
# ---------------------------------------------

# --- CONFIGURATION ---
DATASET_DIR = "dataset"
MODEL_FILE = "injury_model.h5"
MODEL_BRAIN = None # This will hold the loaded AI

# Search terms for missing 'minor' images if we need to download them
MINOR_QUERIES = [
    "small scratch on arm",
    "minor knee scrape",
    "healthy skin arm close up", 
    "healed small scar skin",
    "mosquito bite skin",
    "small paper cut finger"
]

# --- PART 1: AI TRAINING & SETUP LOGIC ---
def download_minor_images():
    print("   ⬇️ Downloading missing 'minor' injury data...")
    temp_dir = "temp_download"
    target_dir = os.path.join(DATASET_DIR, "minor")
    
    os.makedirs(target_dir, exist_ok=True)
    
    for query in MINOR_QUERIES:
        try:
            downloader.download(
                query, 
                limit=20, 
                output_dir=temp_dir, 
                adult_filter_off=True, 
                force_replace=False, 
                timeout=5, 
                verbose=False
            )
        except Exception as e:
            print(f"Error downloading {query}: {e}")

    # Move files
    count = 0
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    shutil.move(os.path.join(root, file), os.path.join(target_dir, f"auto_minor_{count}.jpg"))
                    count += 1
                except: pass
    
    try: shutil.rmtree(temp_dir)
    except: pass
    print(f"   ✅ Added {count} images to 'dataset/minor'.")

def train_model_logic():
    print("   🧠 Training AI Model (this may take 30-60 seconds)...")
    
    # 1. Setup Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, horizontal_flip=True, validation_split=0.2
    )

    try:
        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(224, 224),
            batch_size=16,
            class_mode='binary',
            subset='training'
        )
        
        if train_generator.samples == 0:
            print("   ❌ CRITICAL ERROR: No images found in 'dataset/' folder.")
            return False

        # 2. Build Model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        new_model = Model(inputs=base_model.input, outputs=predictions)
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # 3. Train
        new_model.fit(train_generator, epochs=5, verbose=1)
        
        # 4. Save
        new_model.save(MODEL_FILE)
        print(f"   ✅ Model saved to {MODEL_FILE}")
        return True
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False

def initialize_ai():
    """Checks for model. If missing, downloads data and trains."""
    global MODEL_BRAIN
    
    # 1. Check if model exists
    if os.path.exists(MODEL_FILE):
        print(" * ✅ Found existing AI model.")
    else:
        print(" * ⚠️ Model not found. Starting Auto-Setup...")
        
        # Check Data
        os.makedirs(os.path.join(DATASET_DIR, "severe"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "minor"), exist_ok=True)
        
        minor_count = len(os.listdir(os.path.join(DATASET_DIR, "minor")))
        if minor_count < 10:
            download_minor_images()
            
        # Train
        success = train_model_logic()
        if not success:
            print(" * ❌ AI Setup Failed. App will run without AI features.")
            return

    # 2. Load the model into memory
    print(" * 🚀 Loading Model into Memory...")
    try:
        MODEL_BRAIN = load_model(MODEL_FILE)
        print(" * 🤖 AI Ready!")
    except Exception as e:
        print(f" * ❌ Error loading model: {e}")

# --- PART 2: FLASK WEB ROUTES ---

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/analyze_injury", methods=["POST"])
def analyze_injury():
    if MODEL_BRAIN is None:
        return jsonify({"error": "AI Model is not ready yet. Please wait..."})

    try:
        data = request.get_json()
        image_data = data['image']

        if "base64," in image_data:
            image_data = image_data.split(",")[1]
        
        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))

        processed_image = prepare_image(image, target_size=(224, 224))

        # Predict
        prediction = MODEL_BRAIN.predict(processed_image)[0][0]
        
        # 0 = Minor, 1 = Severe (roughly)
        severity_score = float(prediction)
        is_severe = severity_score > 0.5
        
        confidence_percent = int(severity_score * 100) if is_severe else int((1-severity_score) * 100)

        response = {
            "is_severe": is_severe,
            "confidence": f"{confidence_percent}%",
            "message": "⚠️ POTENTIAL SEVERE INJURY. Go to ER." if is_severe else "✅ Injury appears minor."
        }
        return jsonify(response)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": "Failed to process image"})

@app.route("/submit", methods=["POST"])
def submit():

    raw_patient_data = {
        "full_name": request.form.get("full_name"),
        "sex": request.form.get("sex"),
        "age_range": request.form.get("age_range"),
        "subsymptoms": request.form.getlist("subsymptoms"),
        "additional_info": request.form.get("information_text"),
        "diagnosed_conditions": request.form.get("diagnosed_conditions"),
        "eta": request.form.get("eta_data")
    }

    print("DEBUG: Raw patient data received:", raw_patient_data)  # <--- ADD THIS

    # Call OpenRouter AI
    ai_result = classify_patient(raw_patient_data)

    # Convert AI JSON string into Python dictionary
    triage_analysis = json.loads(ai_result)

    # ONE patient document
    patient_record = {
        "raw_data": raw_patient_data,
        "triage_analysis": triage_analysis,
        "full_name": raw_patient_data["full_name"],
        "created_at": datetime.utcnow()
    }


    patients.insert_one(patient_record)
    return f"Form submitted successfully."

@app.route("/dashboard", methods=["GET"])
def dashboard():
    patients_list = list(patients.find().sort("created_at", -1))
    staff_needed = calculate_staff_needed()

    return render_template(
        "dashboard.html",
        patients=patients_list,
        staff_needed=staff_needed
    )

# --- PART 3: ENTRY POINT ---
if __name__ == "__main__":
    # We run initialization before starting the server
    initialize_ai()
    print("Pre-registration form:")
    print("http://127.0.0.1:5000/")
    print()
    print("ER Dashboard:")
    print("http://127.0.0.1:5000/dashboard")
    print()
    app.run(debug=True, port=5000)