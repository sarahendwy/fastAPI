from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model, Model
from keras.preprocessing import image as keras_image
from flasgger import Swagger, swag_from
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
import base64
import logging
from imghdr import what as detect_image_type

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Deepfake Detection API",
        "description": """
Upload an image to detect whether it is **real** or **fake** using a deep learning model.  
Also returns a **Grad-CAM heatmap** as base64 to visualize model focus.

**Prediction Rule**:  
- If confidence > 0.9 → Real  
- Else → Fake
""",
        "version": "1.0.0",
        "contact": {
            "name": "Sara Hendawy",
            "email": "sarahendawy50@gmail.com",
        },
        "license": {
            "name": "Sadat Academy for Management and Sciences",
            "url": "http://www.sams.edu.eg/en/",
        }
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"],
})

# --- Load Model ---
MODEL_PATH = "model/xpose_pretrained_file.keras"
try:
    logger.info("Attempting to load model from: " + MODEL_PATH)
    model = load_model(MODEL_PATH)
    input_name = model.input[0].name.split(":")[0] if isinstance(model.input, list) else model.input.name.split(":")[0]
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# --- Grad-CAM ---
def get_gradcam_heatmap(model, img_input_dict, original_img_array):
    try:
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input_dict)
            pred_value = predictions[0][0]
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (original_img_array.shape[1], original_img_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img_display = (original_img_array * 255).astype(np.uint8)
        superimposed_img = cv2.addWeighted(img_display, 0.6, heatmap_color, 0.4, 0)

        _, buffer = cv2.imencode(".jpg", superimposed_img)
        heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
        return heatmap_b64, float(pred_value.numpy()), 1 if pred_value > 0.9 else 0
    except Exception as e:
        logger.error(f"Grad-CAM failed: {str(e)}")
        raise Exception(f"Grad-CAM failed: {str(e)}")

# --- Preprocess Image ---
def preprocess_image(img_pil: Image.Image, target_size=(256, 256)) -> tuple:
    img = img_pil.resize(target_size).convert("RGB")
    img_array = keras_image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input, img_array

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Image file to upload (JPEG, PNG, BMP, etc.)'
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction and Grad-CAM Heatmap',
            'examples': {
                'application/json': {
                    "prediction": {
                        "real": 97.23,
                        "fake": 2.77,
                        "label": "Real",
                        "confidence": 0.9723
                    },
                    "heatmap_image_base64": "..."
                }
            }
        },
        400: {'description': 'Invalid input'},
        500: {'description': 'Server error'}
    }
})
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded"}), 400

        file = request.files['file']
        logger.info(f"Received file: {file.filename}")
        contents = file.read()

        image_type = detect_image_type(None, h=contents)
        if image_type not in ["jpeg", "png", "bmp", "gif", "webp"]:
            logger.warning(f"Unsupported image format: {image_type}")
            return jsonify({"detail": f"Unsupported image format: {image_type}"}), 400

        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_input, img_array = preprocess_image(image_pil)
        input_dict = {input_name: img_input} if isinstance(model.input, list) else img_input

        heatmap_b64, pred, class_index = get_gradcam_heatmap(model, input_dict, img_array)

        fake_percent = float((1 - pred) * 100)
        real_percent = float(pred * 100)

        result = {
            "prediction": {
                "real": round(real_percent, 2),
                "fake": round(fake_percent, 2),
                "label": "Real" if pred > 0.9 else "Fake",
                "confidence": round(pred if pred > 0.9 else 1 - pred, 4)
            },
            "heatmap_image_base64": heatmap_b64
        }

        logger.info(f"Prediction completed: {result['prediction']['label']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500

# --- Health Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)
