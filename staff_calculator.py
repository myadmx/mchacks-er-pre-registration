from pymongo import MongoClient
import math

MONGO_URI = "mongodb+srv://er_user:erregistration26@er-registration.g3veiau.mongodb.net/?appName=ER-registration"
client = MongoClient(MONGO_URI)
db = client.er_dashboard
patients = db.patients


def calculate_staff_needed():
    total_weighted_sum = 0

    for patient in patients.find(
        {"triage_analysis.severity": {"$exists": True}},
        {"triage_analysis.severity": 1}
    ):
        severity = patient["triage_analysis"]["severity"]

        if severity in (1, 2):
            weight = 1
        elif severity in (3, 4):
            weight = 2
        elif severity == 5:
            weight = 3

        total_weighted_sum += weight
    return total_weighted_sum


