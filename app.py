import base64
import numpy as np
import io
import os
import shutil
import json
import threading
from flask import Flask, render_template, request, jsonify

# --- NEW IMPORTS (MongoDB & External Classifier) ---
from pymongo import MongoClient
try:
    from classifier import classify_patient
except ImportError:
    print("⚠️ WARNING: 'classifier.py' not found. The /submit route will fail.")
    def classify_patient(data): return '{"error": "Classifier missing"}'

# --- AI & IMAGE IMPORTS ---
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

# --- CONFIGURATION (Image AI) ---
DATASET_DIR = "dataset"
MODEL_FILE = "injury_model.h5"
MODEL_BRAIN = None 

# Expanded search terms to ensure we get enough data
SEARCH_TERMS = {
    "minor": [
        "small scratch on arm", "minor knee scrape", "healthy skin arm close up", 
        "healed small scar skin", "mosquito bite skin", "small paper cut finger"
    ],
    "severe": [
        "severe third degree burn skin", "deep open wound laceration",
        "skin infected wound severe", "severe road rash injury",
        "bloody gash wound leg", "severe skin injury medical",
        "large deep cut on arm", "skin gangrene", "severe laceration wound"
    ]
}

# --- PART 1: IMAGE AI SETUP LOGIC ---
def download_missing_data(category, target_count=100):
    print(f"   ⬇️ Downloading more '{category}' data to reach {target_count} images...")
    temp_dir = f"temp_download_{category}"
    target_dir = os.path.join(DATASET_DIR, category)
    os.makedirs(target_dir, exist_ok=True)
    
    queries = SEARCH_TERMS[category]
    current_count = len(os.listdir(target_dir))
    
    for query in queries:
        if current_count >= target_count: break
        try:
            downloader.download(query, limit=20, output_dir=temp_dir, adult_filter_off=True, force_replace=False, timeout=5, verbose=False)
            
            # Move files
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        shutil.move(os.path.join(root, file), os.path.join(target_dir, f"auto_{category}_{current_count}.jpg"))
                        current_count += 1
        except: pass

    try: shutil.rmtree(temp_dir)
    except: pass
    print(f"   ✅ '{category}' now has {current_count} images.")

def train_model_logic():
    print("   🧠 Training AI Model (this may take 30-60 seconds)...")
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True, validation_split=0.2)
    
    try:
        train_generator = train_datagen.flow_from_directory(
            DATASET_DIR, target_size=(224, 224), batch_size=16, class_mode='binary', subset='training'
        )
        
        # Ensure we found both classes
        if train_generator.samples == 0 or len(train_generator.class_indices) < 2:
            print("   ❌ CRITICAL: Need both 'minor' and 'severe' folders with images.")
            return False

        print(f"   Classes found: {train_generator.class_indices}") # Should be {'minor': 0, 'severe': 1}

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        new_model = Model(inputs=base_model.input, outputs=predictions)
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        new_model.fit(train_generator, epochs=5, verbose=1)
        new_model.save(MODEL_FILE)
        print("   ✅ Model saved!")
        return True
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False

def initialize_ai():
    global MODEL_BRAIN
    
    # 1. ensure folders
    os.makedirs(os.path.join(DATASET_DIR, "severe"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "minor"), exist_ok=True)

    # 2. Check counts & BALANCE THEM
    severe_count = len(os.listdir(os.path.join(DATASET_DIR, "severe")))
    minor_count = len(os.listdir(os.path.join(DATASET_DIR, "minor")))
    
    print(f"   📊 Data Check: {severe_count} Severe | {minor_count} Minor")
    
    # Logic: If one folder is much larger, fill the other one
    needs_training = False
    
    if severe_count < 80: # Aim for at least 80 severe
        download_missing_data("severe", target_count=100)
        needs_training = True
        
    if minor_count < 80:
        download_missing_data("minor", target_count=100)
        needs_training = True

    # 3. Train if needed or if model is missing
    if needs_training or not os.path.exists(MODEL_FILE):
        print("   ⚠️ Starting training sequence...")
        if not train_model_logic():
            print("   ❌ AI Setup Failed.")
            return

    # 4. Load Model
    print(" * 🚀 Loading Model into Memory...")
    try:
        MODEL_BRAIN = load_model(MODEL_FILE)
        print(" * 🤖 Image AI Ready!")
    except Exception as e:
        print(f" * ❌ Error loading model: {e}")

def prepare_image(image, target_size):
    if image.mode != "RGB": image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# --- PART 2: FLASK ROUTES ---

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

@app.route("/analyze_injury", methods=["POST"])
def analyze_injury():
    if MODEL_BRAIN is None:
        # This handles the "AI Model is not loaded" error nicely
        return jsonify({"error": "AI is still starting up. Please wait 10 seconds and try again."})

    try:
        data = request.get_json()
        image_data = data['image']
        if "base64," in image_data: image_data = image_data.split(",")[1]
        
        decoded = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(decoded))
        processed_image = prepare_image(image, target_size=(224, 224))

        prediction = MODEL_BRAIN.predict(processed_image)[0][0]
        severity_score = float(prediction)
        
        # 0 = Minor, 1 = Severe
        is_severe = severity_score > 0.5
        confidence_percent = int(severity_score * 100) if is_severe else int((1-severity_score) * 100)

        return jsonify({
            "is_severe": is_severe,
            "confidence": f"{confidence_percent}%",
            "message": "⚠️ POTENTIAL SEVERE INJURY. Go to ER." if is_severe else "✅ Injury appears minor."
        })
    except Exception as e:
        return jsonify({"error": "Failed to process image"})

@app.route("/submit", methods=["POST"])
def submit():
    symptoms_list = request.form.getlist("subsymptoms")
    symptoms_str = ", ".join(symptoms_list) if symptoms_list else "None explicitly selected"
    
    raw_patient_data = {
        "full_name": request.form.get("name"),
        "phone": request.form.get("phone_number"),
        "medical_card": request.form.get("medical_card"),
        "age": request.form.get("age_range"),
        "symptoms": symptoms_str,
        "other_info": request.form.get("information_text"),
        "diagnosed_conditions": request.form.get("diagnosed_conditions")
    }

    try:
        ai_result = classify_patient(raw_patient_data)
        if isinstance(ai_result, str):
            triage_analysis = json.loads(ai_result)
        else:
            triage_analysis = ai_result
    except Exception as e:
        triage_analysis = {"error": "AI Triage Failed", "details": str(e)}

    document = {
        "raw_data": raw_patient_data,
        "triage_analysis": triage_analysis,
        "status": "waiting",
        "captured_images": request.form.get("captured_images")
    }

    patients.insert_one(document)
    return "Form submitted successfully. Thank you for your time."

if __name__ == "__main__":
    initialize_ai()
    app.run(debug=True, port=5)