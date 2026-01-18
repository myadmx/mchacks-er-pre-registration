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
            
            # Move files and validate them
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        temp_file_path = os.path.join(root, file)
                        target_file_path = os.path.join(target_dir, f"auto_{category}_{current_count}.jpg")
                        
                        try:
                            with Image.open(temp_file_path) as img:
                                img.verify()  # Check if image is valid
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

# --- IMAGE ANALYSIS ROUTE ---
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
        raw_prediction = MODEL_BRAIN.predict(processed_image)[0][0]
        
        # --- MANUAL TYPE FIX (Prevents JSON Error) ---
        score_val = float(raw_prediction)
        
        if score_val > 0.5:
            is_severe_result = True
            confidence_result = int(score_val * 100)
            msg_result = "⚠️ POTENTIAL SEVERE INJURY"
        else:
            is_severe_result = False
            confidence_result = int((1.0 - score_val) * 100)
            msg_result = "✅ Injury appears minor"

        return jsonify({
            "is_severe": is_severe_result,
            "confidence": f"{confidence_result}%",
            "message": msg_result
        })
        
    except Exception as e:
        print(f"Error in analyze_injury: {e}") 
        return jsonify({"error": f"Server Error: {str(e)}"})

# --- FORM SUBMISSION ROUTE ---
@app.route("/submit", methods=["POST"])
def submit():
    # 1. Get the name correctly from HTML name="name"
    full_name = request.form.get("name") 
    
    # 2. Join symptoms list into a string
    subsymptoms_list = request.form.getlist("subsymptoms")
    subsymptoms_str = ", ".join(subsymptoms_list) if subsymptoms_list else "None selected"

    # 3. Prepare data for AI
    raw_patient_data = {
        "full_name": full_name,
        "sex": request.form.get("sex"),
        "age": request.form.get("age_range"),
        "subsymptoms": subsymptoms_str,       
        "additional_info": request.form.get("information_text"),
        "diagnosed_conditions": request.form.get("diagnosed_conditions"),
        "eta": request.form.get("eta_data"),
        "captured_images": request.form.get("captured_images")
    }

    print(f"DEBUG: Processing patient: {full_name}")
    print(f"DEBUG: Symptoms sending to AI: {subsymptoms_str}")

    try:
        # 4. Call Classifier
        ai_result = classify_patient(raw_patient_data)
        
        if isinstance(ai_result, str):
            clean_json = ai_result.replace("```json", "").replace("```", "").strip()
            triage_analysis = json.loads(clean_json)
        else:
            triage_analysis = ai_result
            
    except Exception as e:
        print(f"AI Error: {e}")
        # Fallback if AI fails
        triage_analysis = {
            "full_name": full_name,
            "main_symptoms": ["AI Error"],
            "main_subsymptoms": subsymptoms_list, # Use raw list if AI fails
            "severity": 1,
            "triage_category": "Non-critical"
        }

    # 5. Save to Database
    patient_record = {
        "full_name": full_name,
        "raw_data": raw_patient_data,
        "triage_analysis": triage_analysis,
        "created_at": datetime.utcnow(),
        "status": "waiting"
    }

    patients.insert_one(patient_record)
    return "Form submitted successfully. You can close this window."

@app.route("/dashboard", methods=["GET"])
def dashboard():
    patients_list = list(patients.find().sort("created_at", -1))
    staff_needed = calculate_staff_needed()
    return render_template("dashboard.html", patients=patients_list, staff_needed=staff_needed)

if __name__ == "__main__":
    initialize_ai()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)