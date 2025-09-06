# # app.py

# import os
# import base64
# import uuid
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from haversine import haversine, Unit
# from deepface import DeepFace

# # --- Flask App Initialization ---
# app = Flask(__name__)
# # CORS is needed to allow communication between the server and the frontend.
# # It's good practice to restrict origins in a production environment.
# CORS(app)

# # --- Configuration & Data ---
# # In a real app, this data would come from a database.
# CLASS_LOCATIONS = {
#     "CS101": {
#         "name": "Computer Science Building, Room 101",
#         "lat": 16.7953091,
#         "lon": 80.8228997
#     },
#     "PHY203": {
#         "name": "Physics Hall, Room 203",
#         "lat": 40.712776,
#         "lon": -74.005974
#     }
# }
# # The allowed distance in meters from the class location.
# # Note: 9 meters is a very strict radius.
# LOCATION_RADIUS_METERS = 900000000

# # The path to the reference image for face verification.
# # Ensure 'reference.jpg' is in the same directory as this script.
# REFERENCE_IMG_PATH = "WIN_20250906_10_34_24_Pro.jpg"
# if not os.path.exists(REFERENCE_IMG_PATH):
#     raise FileNotFoundError(f"Reference image not found at '{REFERENCE_IMG_PATH}'")

# # --- Route for serving the frontend ---
# @app.route('/')
# def index():
#     """Renders the main HTML page."""
#     return render_template('index.html')

# # --- API Endpoint for Location Verification ---
# @app.route('/verify-location', methods=['POST'])
# def handle_verify_location():
#     """
#     Verifies if the user's coordinates are within the allowed radius for a class.
#     """
#     data = request.json
#     class_id = data.get('class_id')
#     user_lat = data.get('latitude')
#     user_lon = data.get('longitude')

#     if not all([class_id, user_lat, user_lon]):
#         return jsonify({"status": "error", "message": "Missing class_id, latitude, or longitude."}), 400

#     class_info = CLASS_LOCATIONS.get(class_id)
#     if not class_info:
#         return jsonify({"status": "error", "message": f"Class ID '{class_id}' not found."}), 404

#     # Calculate distance
#     class_location = (class_info["lat"], class_info["lon"])
#     user_location = (float(user_lat), float(user_lon))
#     distance = haversine(class_location, user_location, unit=Unit.METERS)

#     if distance <= LOCATION_RADIUS_METERS:
#         print(f"SUCCESS: Location verified for '{class_id}'. Distance: {distance:.2f}m.")
#         return jsonify({
#             "status": "success",
#             "message": f"Location verified. You are {distance:.2f} meters from the class."
#         })
#     else:
#         print(f"FAILURE: User is too far for '{class_id}'. Distance: {distance:.2f}m.")
#         return jsonify({
#             "status": "failure",
#             "message": f"Attendance denied. You are {distance:.2f} meters away from the class location."
#         })

# # --- API Endpoint for Face Verification ---
# @app.route('/verify-face', methods=['POST'])
# def handle_verify_face():
#     """
#     Verifies the captured face image against the reference image.
#     Expects a base64 encoded image string.
#     """
#     data = request.json
#     image_data_url = data.get('image')

#     if not image_data_url:
#         return jsonify({"status": "error", "message": "No image data received."}), 400

#     try:
#         # Decode the base64 image
#         header, encoded = image_data_url.split(",", 1)
#         image_bytes = base64.b64decode(encoded)

#         # Save to a temporary file for DeepFace to process
#         captured_img_path = f"{uuid.uuid4()}.jpg"
#         with open(captured_img_path, "wb") as f:
#             f.write(image_bytes)

#         # Perform face verification
#         result = DeepFace.verify(
#             img1_path=REFERENCE_IMG_PATH,
#             img2_path=captured_img_path,
#             model_name="Facenet",
#             enforce_detection=False # Be more lenient if a face isn't perfectly centered
#         )

#         # Clean up the temporary file
#         os.remove(captured_img_path)

#         if result["verified"]:
#             print("SUCCESS: Face verification successful.")
#             return jsonify({"status": "success", "message": "Face verified successfully."})
#         else:
#             print("FAILURE: Face verification failed.")
#             return jsonify({"status": "failure", "message": "Face does not match. Attendance denied."})

#     except Exception as e:
#         print(f"An error occurred during face verification: {e}")
#         # If a temp file was created before the error, try to remove it
#         if 'captured_img_path' in locals() and os.path.exists(captured_img_path):
#             os.remove(captured_img_path)
#         return jsonify({"status": "error", "message": "Could not process the image. Please try again."}), 500

# # --- Main Execution ---
# if __name__ == '__main__':
#     # Use host='0.0.0.0' to make it accessible on your local network
#     app.run(host='0.0.0.0', port=5000, debug=True)

# app.py

import os
import base64
import uuid

# --- MEMORY OPTIMIZATION FOR TENSORFLOW (THE FIX) ---
# These lines MUST be at the top, before importing deepface
import tensorflow as tf
# Configure TensorFlow to be less memory-intensive by using single-threaded operations.
# This is crucial for free-tier deployment environments.
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# ---------------------------------------------------

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from haversine import haversine, Unit
from deepface import DeepFace

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration & Data ---
CLASS_LOCATIONS = {
    "CS101": {
        "name": "Computer Science Building, Room 101",
        "lat": 16.7953091,
        "lon": 80.8228997
    }
}
# Using a larger radius for testing based on your previous logs. Adjust as needed.
LOCATION_RADIUS_METERS = 38061 

REFERENCE_IMG_PATH = "reference.jpg"
if not os.path.exists(REFERENCE_IMG_PATH):
    print(f"WARNING: Reference image not found at '{REFERENCE_IMG_PATH}'.")

# --- MODEL WARM-UP FUNCTION ---
def load_initial_model():
    """
    Forces DeepFace to download and load the model into memory once when the
    server starts, preventing timeouts on the first user request.
    """
    print("Pre-loading FaceNet model...")
    try:
        _ = DeepFace.represent(img_path=REFERENCE_IMG_PATH, model_name="Facenet", enforce_detection=False)
        print("FaceNet model pre-loaded successfully.")
    except Exception as e:
        print(f"Error pre-loading model: Could not find face in reference image or another error occurred. {e}")
        # The app will still run, but the first /verify-face call might be slow.

# Call the warm-up function when the application starts
load_initial_model()

# --- Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

# (The rest of your app.py file remains exactly the same...)
# ... @app.route('/verify-location') ...
# ... @app.route('/verify-face') ...
# ... if __name__ == '__main__': ...

@app.route('/verify-location', methods=['POST'])
def handle_verify_location():
    """Verifies user's coordinates are within the allowed radius."""
    data = request.json
    class_id = data.get('class_id')
    user_lat = data.get('latitude')
    user_lon = data.get('longitude')

    if not all([class_id, user_lat, user_lon]):
        return jsonify({"status": "error", "message": "Missing required location data."}), 400

    class_info = CLASS_LOCATIONS.get(class_id)
    if not class_info:
        return jsonify({"status": "error", "message": f"Class ID '{class_id}' not found."}), 404

    class_location = (class_info["lat"], class_info["lon"])
    user_location = (float(user_lat), float(user_lon))
    distance = haversine(class_location, user_location, unit=Unit.METERS)

    if distance <= LOCATION_RADIUS_METERS:
        return jsonify({
            "status": "success",
            "message": f"Location verified. You are {distance:.2f} meters from the class."
        })
    else:
        return jsonify({
            "status": "failure",
            "message": f"Attendance denied. You are {distance:.2f} meters away."
        })

@app.route('/verify-face', methods=['POST'])
def handle_verify_face():
    """Verifies a captured face image against the reference image."""
    data = request.json
    image_data_url = data.get('image')

    if not image_data_url:
        return jsonify({"status": "error", "message": "No image data received."}), 400

    if not os.path.exists(REFERENCE_IMG_PATH):
         return jsonify({"status": "error", "message": "Server error: Reference image is missing."}), 500

    captured_img_path = ""
    try:
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        captured_img_path = f"{uuid.uuid4()}.jpg"
        with open(captured_img_path, "wb") as f:
            f.write(image_bytes)

        result = DeepFace.verify(
            img1_path=REFERENCE_IMG_PATH,
            img2_path=captured_img_path,
            model_name="Facenet",
            enforce_detection=False
        )

        if result["verified"]:
            return jsonify({"status": "success", "message": "Face verified successfully."})
        else:
            return jsonify({"status": "failure", "message": "Face does not match. Attendance denied."})

    except Exception as e:
        print(f"An error occurred during face verification: {e}")
        return jsonify({"status": "error", "message": "Could not process the image."}), 500
    finally:
        if os.path.exists(captured_img_path):
            os.remove(captured_img_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)