
from pathlib import Path
import re

import av
import cv2
import numpy as np
import pytesseract
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_resnet18_fp16.pth"

TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

CLASS_NAMES = [
    "play_100",
    "play_200",
    "play_500",
    "real_100",
    "real_200",
    "real_500",
]

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #17365d;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    font-size: 17px;
    color: #526579;
    margin-bottom: 30px;
}
.feature-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #e1e8f0;
    min-height: 125px;
}
.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #dce5ef;
    margin-top: 20px;
}
.genuine {
    color: #16834a;
    font-size: 28px;
    font-weight: bold;
}
.play-money {
    color: #c0392b;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES
    )

    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Class mismatch. Model classes: {class_names}"
        )

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# 4. OCR IMAGE PREPROCESSING
# =========================================================

def normalize_ocr_text(text):

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


def contains_children_bank_marker(text):

    normalized = normalize_ocr_text(text)

    return (
        "CHILDREN" in normalized
        and (
            "BANK" in normalized
            or "BAN" in normalized
        )
    )


def preprocess_ocr_image(image_rgb):

    height, width = image_rgb.shape[:2]

    # Resize according to image size, not fixed note position.
    if width < 1200:
        scale = min(3.0, 1200 / max(width, 1))
    else:
        scale = 1.0

    if scale > 1:
        image_rgb = cv2.resize(
            image_rgb,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Threshold processing.
    threshold = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Adaptive threshold processing.
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return gray, enhanced, threshold, adaptive


# =========================================================
# 5. OCR ON THE COMPLETE IMAGE
# =========================================================

def scan_complete_image(image_rgb):

    gray, enhanced, threshold, adaptive = (
        preprocess_ocr_image(image_rgb)
    )

    processed_images = [
        ("Gray image", gray),
        ("Enhanced image", enhanced),
        ("Otsu threshold", threshold),
        ("Adaptive threshold", adaptive)
    ]

    all_text = []

    for image_name, processed in processed_images:

        for page_mode in [6, 11]:

            text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config=f"--oem 3 --psm {page_mode}"
            )

            all_text.append(
                f"{image_name}, PSM {page_mode}:\n{text}"
            )

            if contains_children_bank_marker(text):
                return True, "\n\n".join(all_text)

    return False, "\n\n".join(all_text)


# =========================================================
# 6. ORIGINAL OCR CROP AS AN EXTRA CHECK
# =========================================================

def read_original_crop(image_rgb):

    height, width = image_rgb.shape[:2]

    crop = image_rgb[
        int(height * 0.43):int(height * 0.55),
        int(width * 0.59):int(width * 0.87)
    ]

    if crop.size == 0:
        return ""

    crop = cv2.resize(
        crop,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    original_text = pytesseract.image_to_string(
        gray,
        lang="eng",
        config="--psm 6"
    )

    threshold_text = pytesseract.image_to_string(
        threshold,
        lang="eng",
        config="--psm 6"
    )

    return (
        "Original crop:\n"
        + original_text
        + "\nThreshold crop:\n"
        + threshold_text
    )


# =========================================================
# 7. CHILDREN'S BANK DETECTION
# =========================================================

def detect_children_bank_text(image_rgb):

    try:

        orientations = [
            ("Original", image_rgb),
            ("Horizontal flip", cv2.flip(image_rgb, 1)),
            (
                "180 degree rotation",
                cv2.rotate(image_rgb, cv2.ROTATE_180)
            ),
            (
                "Rotated and flipped",
                cv2.flip(
                    cv2.rotate(
                        image_rgb,
                        cv2.ROTATE_180
                    ),
                    1
                )
            )
        ]

        all_results = []

        # First scan the entire image in different orientations.
        for orientation_name, oriented_image in orientations:

            found, extracted_text = scan_complete_image(
                oriented_image
            )

            all_results.append(
                f"\n========== {orientation_name} ==========\n"
                + extracted_text
            )

            if found:
                return True, "\n".join(all_results)

        # Retain the original working crop as an extra check.
        crop_text = read_original_crop(image_rgb)

        all_results.append(
            "\n========== ORIGINAL WORKING CROP ==========\n"
            + crop_text
        )

        if contains_children_bank_marker(crop_text):
            return True, "\n".join(all_results)

        return False, "\n".join(all_results)

    except Exception as error:

        return False, f"OCR error: {error}"


# =========================================================
# 8. COLOUR ANALYSIS
# =========================================================

def analyze_colour(image_rgb):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    hsv = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2HSV
    )

    return {
        "hue": round(float(np.mean(hsv[:, :, 0])), 2),
        "saturation": round(float(np.mean(hsv[:, :, 1])), 2),
        "brightness": round(float(np.mean(hsv[:, :, 2])), 2)
    }


# =========================================================
# 9. PREDICTION FUNCTION
# =========================================================

def predict_currency(image_rgb, model):

    pil_image = Image.fromarray(
        image_rgb
    ).convert("RGB")

    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    model_class = CLASS_NAMES[predicted_index.item()]
    model_confidence = float(confidence.item()) * 100

    marker_found, extracted_text = (
        detect_children_bank_text(image_rgb)
    )

    colour_data = analyze_colour(image_rgb)

    if marker_found:

        final_class = "play_money_detected"
        final_confidence = 100.0

    else:

        final_class = model_class
        final_confidence = model_confidence

    return {
        "final_class": final_class,
        "confidence": final_confidence,
        "model_class": model_class,
        "model_confidence": model_confidence,
        "marker_found": marker_found,
        "ocr_text": extracted_text,
        "colour_data": colour_data
    }


# =========================================================
# 10. DISPLAY RESULT
# =========================================================

def show_result(result):

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    if result["final_class"] == "play_money_detected":

        st.markdown(
            '<p class="play-money">⚠️ PLAY MONEY DETECTED</p>',
            unsafe_allow_html=True
        )

        st.write(
            "The OCR system detected a Children's Bank text marker."
        )

    elif result["final_class"].startswith("play_"):

        denomination = result["final_class"].replace(
            "play_", "₹"
        )

        st.markdown(
            '<p class="play-money">⚠️ PLAY MONEY</p>',
            unsafe_allow_html=True
        )

        st.write(f"Predicted category: {denomination}")

    else:

        denomination = result["final_class"].replace(
            "real_", "₹"
        )

        st.markdown(
            '<p class="genuine">✓ MODEL CLASSIFICATION: GENUINE CATEGORY</p>',
            unsafe_allow_html=True
        )

        st.write(f"Predicted category: {denomination}")

    st.write(
        f"Final confidence: {result['confidence']:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View OCR and model details"):

        st.write(
            "OCR marker detected:",
            "Yes" if result["marker_found"] else "No"
        )

        st.write("Extracted OCR text:")
        st.code(result["ocr_text"] or "No text detected")

        st.write("Model prediction:", result["model_class"])

        st.write(
            f"Model confidence: {result['model_confidence']:.2f}%"
        )

        st.write("Colour analysis:")
        st.json(result["colour_data"])

    st.caption(
        "Educational image-classification project. "
        "This is not an official currency-authentication decision."
    )


# =========================================================
# 11. CAMERA PROCESSOR
# =========================================================

class CurrencyCamera(VideoProcessorBase):

    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.latest_frame = img.copy()

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================================================
# 12. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">💵 FAKE CURRENCY DETECTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Currency Image Classification System'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        '<div class="feature-card">'
        '<h3>📷 Live Camera</h3>'
        '<p>Capture currency using your camera</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div class="feature-card">'
        '<h3>🖼️ Image Upload</h3>'
        '<p>Upload a currency image for analysis</p>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        '<div class="feature-card">'
        '<h3>🤖 AI Detection</h3>'
        '<p>ResNet18 and OCR-based analysis</p>'
        '</div>',
        unsafe_allow_html=True
    )

st.write("")


# =========================================================
# 13. LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as error:

    st.error(f"Unable to load model: {error}")
    st.stop()


# =========================================================
# 14. INPUT OPTIONS
# =========================================================

st.markdown("## Choose Detection Method")

mode = st.radio(
    "Select an option:",
    ["Upload Image", "Live Camera"],
    horizontal=True
)


# =========================================================
# 15. UPLOAD IMAGE
# =========================================================

if mode == "Upload Image":

    st.markdown("### Upload Currency Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        st.image(
            image_rgb,
            caption="Uploaded Currency Image",
            width="stretch"
        )

        if st.button(
            "🔍 TEST UPLOADED IMAGE",
            width="stretch"
        ):

            with st.spinner("Analyzing currency image..."):

                result = predict_currency(
                    image_rgb,
                    model
                )

                st.session_state["prediction_result"] = result

        if "prediction_result" in st.session_state:

            show_result(
                st.session_state["prediction_result"]
            )


# =========================================================
# 16. LIVE CAMERA
# =========================================================

elif mode == "Live Camera":

    st.markdown("### Live Currency Detection")

    st.write(
        "Start your camera, place the currency in front of it, "
        "and capture an image."
    )

    ctx = webrtc_streamer(
        key="fake-currency-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=CurrencyCamera,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True,
    )

    if ctx.video_processor:

        if st.button(
            "📸 CAPTURE IMAGE",
            width="stretch"
        ):

            frame = ctx.video_processor.latest_frame

            if frame is not None:

                captured_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                captured_rgb = np.ascontiguousarray(
                    captured_rgb
                )

                cv2.imwrite(
                    str(BASE_DIR / "live_capture_debug.jpg"),
                    cv2.cvtColor(
                        captured_rgb,
                        cv2.COLOR_RGB2BGR
                    )
                )

                st.session_state["captured_image"] = captured_rgb

                st.session_state.pop(
                    "camera_result",
                    None
                )

            else:

                st.warning(
                    "Camera frame not available. Please wait and try again."
                )

    if "captured_image" in st.session_state:

        st.markdown("### Captured Image")

        st.image(
            st.session_state["captured_image"],
            caption="Captured Currency",
            width="stretch"
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "🔍 TEST CAPTURED IMAGE",
                width="stretch"
            ):

                with st.spinner("Analyzing captured image..."):

                    st.session_state["camera_result"] = (
                        predict_currency(
                            st.session_state["captured_image"],
                            model
                        )
                    )

        with col2:

            if st.button(
                "🔄 RETAKE IMAGE",
                width="stretch"
            ):

                st.session_state.pop(
                    "captured_image",
                    None
                )

                st.session_state.pop(
                    "camera_result",
                    None
                )

                st.rerun()

        if "camera_result" in st.session_state:

            show_result(
                st.session_state["camera_result"]
            )


# =========================================================
# 17. FOOTER
# =========================================================

st.write("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;">
        <p>Fake Currency Detection | Machine Learning and Computer Vision Project</p>
        <p>ResNet18 • OCR • OpenCV • Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
