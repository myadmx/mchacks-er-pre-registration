import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def classify_patient(patient_data):
    """
    Receives raw patient form data and returns structured triage classification.
    """

    prompt = f"""
You are an emergency department triage assistant.

You will receive patient pre-registration data.
Your task is to analyze the medical information and return a structured triage assessment.

Rules:
- Full name and Sex MUST be returned exactly as provided.
- Identify the most important main symptoms and subsymptoms.
- Severity is an integer from 1 to 5 (1 = not severe, 5 = extremely severe).
- Triage Category must be ONE of:
  - Non-critical
  - Critical
  - Life-threatening
- Base your decision on standard ER triage reasoning.
- Respond ONLY with valid JSON. No explanations.

Patient data:
Full name: {patient_data.get("full_name")}
Sex: {patient_data.get("sex")}
Age: {patient_data.get("age")}
Symptoms description: {patient_data.get("symptoms")}

Required JSON format:
{{
  "full_name": "",
  "sex": "",
  "main_symptoms": [],
  "main_subsymptoms": [],
  "severity": 1,
  "triage_category": ""
}}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    result = response.json()
    ai_output = result["choices"][0]["message"]["content"]

    return ai_output
