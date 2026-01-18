import base64
import numpy as np
import io
import os
import shutil
import json
import threading
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from pymongo import MongoClient

# --- CUSTOM MODULE IMPORTS ---
# Ensure classifier.py and staff_calculator.py are in the same folder
try:
    from classifier import classify_patient
except ImportError:
    print("⚠️ WARNING: 'classifier.py' not found. Text triage will return errors.")
    def classify_patient(data): return '{"error": "Classifier missing"}'

try:
    from staff_calculator import calculate_staff_needed
except ImportError:
    print("⚠️ WARNING: 'staff_calculator.py' not found. Staff calculation disabled.")
    def calculate_staff_needed(): return 0

# --- AI & IMAGE LIB IMPORTS ---
from PIL import Image
from bing_image_downloader import downloader
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================
MONGO_URI = "mongodb+srv://er_user:erregistration26@er-registration.g3veiau.mongodb.net/?appName=ER-registration"
client = MongoClient(MONGO_URI)
db = client.er_dashboard
patients = db.patients

# ==========================================
# 2. IMAGE AI CONFIGURATION & LOGIC
# ==========================================
DATASET_DIR = "dataset"
MODEL_FILE = "injury_model.h5"
MODEL_BRAIN = None 

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
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        temp_file_path = os.path.join(root, file)
                        target_file_path = os.path.join(target_dir, f"auto_{category}_{current_count}.jpg")
                        try:
                            with Image.open(temp_file_path) as img:
                                img.verify() 
                            shutil.move(temp_file_path, target_file_path)
                            current_count += 1
                        except:
                            pass
        except Exception as e:
            print(f"   ❌ Download error: {e}")
    try: shutil.rmtree(temp_dir)
    except: pass

def train_model_logic():
    print("   🧠 Training AI Model (this may take 30-60 seconds)...")
    try:
        valid_images = []
        labels = []
        for category in ['minor', 'severe']:
            category_dir = os.path.join(DATASET_DIR, category)
            label = 0 if category == 'minor' else 1
            if os.path.exists(category_dir):
                for file in os.listdir(category_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img_path = os.path.join(category_dir, file)
                            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                            img_array = tf.keras.preprocessing.image.img_to_array(img)
                            valid_images.append(img_array)
                            labels.append(label)
                        except: pass
        
        if len(valid_images) < 10:
            print("   ❌ Not enough valid images to train.")
            return False
            
        X = np.array(valid_images)
        y = np.array(labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        new_model = Model(inputs=base_model.input, outputs=predictions)
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        new_model.fit(train_datagen.flow(X_train, y_train, batch_size=16), validation_data=val_datagen.flow(X_val, y_val, batch_size=16), epochs=5, verbose=1)
        new_model.save(MODEL_FILE)
        return True
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return False

def initialize_ai():
    global MODEL_BRAIN
    os.makedirs(os.path.join(DATASET_DIR, "severe"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "minor"), exist_ok=True)
    
    if not os.path.exists(MODEL_FILE):
        download_missing_data("severe", 80)
        download_missing_data("minor", 80)
        train_model_logic()

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

# ==========================================
# 3. FLASK ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def form():
    return render_template("form.html")

# --- IMAGE ANALYSIS ROUTE (Used by the 'Check Severity' button) ---
@app.route("/analyze_injury", methods=["POST"])
def analyze_injury():
    if MODEL_BRAIN is None:
        return jsonify({"error": "AI is initializing. Please wait."})

    try:
        data = request.get_json()
        image_data = data.get('image', '')
        if "base64," in image_data: image_data = image_data.split(",")[1]
        
        try:
            decoded = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded))
        except:
            return jsonify({"error": "Invalid image data."})
        
        processed_image = prepare_image(image, target_size=(224, 224))
        prediction = MODEL_BRAIN.predict(processed_image)[0][0]
        
        is_severe = prediction > 0.5
        confidence = int(prediction * 100) if is_severe else int((1-prediction) * 100)

        return jsonify({
            "is_severe": is_severe,
            "confidence": f"{confidence}%",
            "message": "⚠️ POTENTIAL SEVERE INJURY" if is_severe else "✅ Injury appears minor"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# --- FORM SUBMISSION ROUTE ---
@app.route("/submit", methods=["POST"])
def submit():
    # 1. Gather all data from the updated HTML form
    full_name = request.form.get("name")  # HTML name="name"
    raw_patient_data = {
        "full_name": full_name,
        "sex": request.form.get("sex"),
        "medical_card": request.form.get("medical_card"),
        "phone_number": request.form.get("phone_number"),
        "age_range": request.form.get("age_range"),
        "subsymptoms": request.form.getlist("subsymptoms"), # List of checked boxes
        "additional_info": request.form.get("information_text"),
        "diagnosed_conditions": request.form.get("diagnosed_conditions"),
        "eta": request.form.get("eta_data"),
        "captured_images": request.form.get("captured_images") # JSON string of base64s
    }

    print("DEBUG: Processing patient:", full_name)

    # 2. Call Text AI for Triage
    try:
        ai_result = classify_patient(raw_patient_data)
        if isinstance(ai_result, str):
            triage_analysis = json.loads(ai_result)
        else:
            triage_analysis = ai_result
    except Exception as e:
        print(f"AI Error: {e}")
        triage_analysis = {"error": "AI Triage Failed", "details": str(e)}

    # 3. Create Patient Record
    patient_record = {
        "full_name": full_name,
        "raw_data": raw_patient_data,
        "triage_analysis": triage_analysis,
        "created_at": datetime.utcnow(), # Important for sorting in Dashboard
        "status": "waiting"
    }

    # 4. Insert into MongoDB
    patients.insert_one(patient_record)

    return "Form submitted successfully. You can close this window."

# --- DASHBOARD ROUTE ---
@app.route("/dashboard", methods=["GET"])
def dashboard():
    # Sort by newest first
    patients_list = list(patients.find().sort("created_at", -1))
    
    # Calculate staff needed based on current patient load
    staff_needed = calculate_staff_needed()

    return render_template(
        "dashboard.html",
        patients=patients_list,
        staff_needed=staff_needed
    )

if __name__ == "__main__":
    # Initialize the Image AI before starting the server
    initialize_ai()
    
    print("\n✅ App is running!")
    print("   - Form:      http://127.0.0.1:5000/")
    print("   - Dashboard: http://127.0.0.1:5000/dashboard\n")
    
    app.run(debug=True, port=5000)