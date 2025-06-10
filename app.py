from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from keras.models import load_model, Model
from keras.preprocessing import image as keras_image
from pydantic import BaseModel
from typing import Literal
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
import base64
from imghdr import what as detect_image_type
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI Metadata ---
app = FastAPI(
    title="Deepfake Detection API",
    description="""
Upload an image to detect whether it is **real** or **fake** using a deep learning model.  
Also returns a **Grad-CAM heatmap** as base64 to visualize model focus.

**Prediction Rule**:  
- If confidence > 0.9 → Real  
- Else → Fake
""",
    version="1.0.0",
    contact={
        "name": "Mahmoud Elkholy",
        "email": "deepfake017@gmail.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# --- CORS for Local Frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- Load Model ---
MODEL_PATH = "model/xpose_pretrained_file.keras"
try:
    model = load_model(MODEL_PATH)
    input_name = model.input[0].name.split(":")[0] if isinstance(model.input, list) else model.input.name.split(":")[0]
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise Exception(f"Failed to load model: {str(e)}")

# --- Grad-CAM Heatmap Function ---
def get_gradcam_heatmap(model, img_input_dict, original_img_array):
    try:
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
        grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input_dict)
            pred_value = predictions[0][0]
            class_index = 1 if pred_value > 0.9 else 0
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
        return heatmap_b64, float(pred_value.numpy()), class_index  # Convert to Python float
    except Exception as e:
        logger.error(f"Grad-CAM failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {str(e)}")

# --- Preprocess Image ---
def preprocess_image(img_pil: Image.Image, target_size=(256, 256)) -> tuple:
    """
    Preprocess image to match model input (256x256, normalize to [0,1]).
    """
    img = img_pil.resize(target_size).convert("RGB")
    img_array = keras_image.img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img_input, img_array

# --- Response Model ---
class PredictionResult(BaseModel):
    real: float
    fake: float
    label: Literal["Real", "Fake"]
    confidence: float

class PredictionResponse(BaseModel):
    prediction: PredictionResult
    heatmap_image_base64: str

# --- Predict Endpoint ---
@app.post(
    "/predict",
    summary="Predict Real or Fake from Image",
    description="Upload an image to get real/fake classification and a Grad-CAM heatmap.",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts: JPEG, PNG, BMP, GIF, WEBP images
    Returns: Confidence scores (real vs fake) + heatmap
    """
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        image_type = detect_image_type(None, h=contents)
        if image_type not in ["jpeg", "png", "bmp", "gif", "webp"]:
            logger.warning(f"Unsupported image format: {image_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {image_type}")

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
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- Health Check Endpoint ----
@app.get("/health", summary="Check API Health")
async def health_check():
    return {"status": "API is running"}