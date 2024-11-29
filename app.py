from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib  # Use joblib for loading the model
import logging

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
MODEL_PATH = "hybrid_model.joblib"  # Updated file extension to .joblib
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    model = None  # Set to None to handle later in endpoints


@app.route("/")
def home():
    # Render the HTML page
    return render_template("/index.html")  # Ensure index.html is in the 'templates' folder


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded properly. Check server logs for details."}), 500

        # Extract form data from the request
        data = request.form.to_dict()
        logging.debug("Received form data: %s", data)

        # Define required features in the correct order
        required_features = [
            'Hole (Nos)', 'Depth (m)', 'Spacing(m)',
            'Burden (m)', 'Stemming(m)', 'Decking(m)', 'Total Drill (RMT)',
            'Explosive(kg)', 'Volume(m3)', 'Powder Factor(kg/m3)',
            'Av. CPH', 'MCPD (kg/D)', 'Seis. Dist. (m)'
        ]

        # Handle missing or "Nil" values by replacing with 0
        processed_data = {}
        for feature in required_features:
            value = data.get(feature, "").strip()
            if value.lower() == "nil" or value == "":
                processed_data[feature] = 0.0  # Replace "Nil" or empty with 0
            else:
                processed_data[feature] = float(value)  # Convert to float

        # Convert processed data to a DataFrame
        input_data = pd.DataFrame([processed_data])
        logging.debug("Processed input data: %s", input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]
        logging.info("Prediction successful: %s", prediction)

        # Return the prediction as a JSON response
        return jsonify({"predicted_ppv": prediction})

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8080)
