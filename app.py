from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import threading
import time
import pandas as pd
import joblib
import os

# ==============================
# Load trained model
# ==============================
try:
    model = joblib.load("crop_model.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# Flask app
app = Flask(__name__)
CORS(app)

# ==============================
# Weather API function
# ==============================
def get_weather(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10).json()
        if response.get("cod") != 200:
            return None, response.get("message", "Unknown error")

        temperature = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        return (temperature, humidity), None
    except Exception as e:
        return None, str(e)

# ==============================
# API Endpoint
# ==============================
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # --- Weather Handling ---
        weather, error = None, None
        api_key = data.get("api_key") or os.environ.get("OPENWEATHER_API_KEY")
        if "city" in data and api_key:
            weather, error = get_weather(data["city"], api_key)

        if not weather:
            # Fallback: if temperature & humidity given manually
            if "temperature" in data and "humidity" in data:
                temperature = data["temperature"]
                humidity = data["humidity"]
            else:
                return jsonify({"error": f"Weather API failed: {error}. "
                                         f"Provide 'temperature' and 'humidity' manually."}), 500
        else:
            temperature, humidity = weather

        # --- Input for ML model ---
        try:
            input_data = pd.DataFrame([[
                data["N"], data["P"], data["K"],
                temperature, humidity,
                data["ph"], data["rainfall"]
            ]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        except KeyError as ke:
            return jsonify({"error": f"Missing required field: {str(ke)}"}), 400

        # --- Prediction ---
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "city": data.get("city"),
            "weather": {"temperature": temperature, "humidity": humidity},
            "soil": {
                "N": data["N"], "P": data["P"], "K": data["K"],
                "ph": data["ph"], "rainfall": data["rainfall"]
            },
            "prediction": prediction,
            "top3": [
                {"crop": crop, "probability": round(prob * 100, 2)} for crop, prob in top3
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Optional home / health routes
# ==============================
@app.route("/")
def home():
    return jsonify({"message": "🌾 Crop Recommendation API is running!"})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ==============================
# Run Flask on Render
# ==============================
def keep_alive_ping(interval=300):
    """Periodically ping the /health endpoint to keep the server alive."""
    while True:
        try:
            requests.get(f"http://localhost:{os.environ.get('PORT', 5000)}/health", timeout=5)
        except Exception:
            pass
        time.sleep(interval)

if __name__ == "__main__":
    # Start keep-alive thread
    threading.Thread(target=keep_alive_ping, args=(300,), daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
