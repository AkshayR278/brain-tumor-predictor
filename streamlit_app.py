from pathlib import Path
import io
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="NeuroVision | MRI Review",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.keras"
MODEL_LABELS_PATH = APP_DIR / "model_labels.json"
EXPECTED_CLASS_LABELS = {"glioma", "meningioma", "notumor", "pituitary"}
DISPLAY_LABELS = {
    "pituitary": "Pituitary",
    "glioma": "Glioma",
    "notumor": "No tumor",
    "meningioma": "Meningioma",
}


@st.cache_resource
def load_model_cached() -> tuple[tf.keras.Model, tuple[str, ...]]:
    if not MODEL_PATH.is_file() or not MODEL_LABELS_PATH.is_file():
        raise RuntimeError(
            "The trained model files are missing. Run "
            "`python train_model.py` or execute all cells in "
            "`brain_tumour_detection.ipynb` first."
        )
    try:
        metadata = json.loads(MODEL_LABELS_PATH.read_text(encoding="utf-8"))
        class_labels = tuple(metadata["class_names"])
        if (
            len(class_labels) != len(EXPECTED_CLASS_LABELS)
            or set(class_labels) != EXPECTED_CLASS_LABELS
        ):
            raise ValueError(f"Unexpected model class names: {class_labels}")

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        if model.input_shape != (None, 128, 128, 3):
            raise ValueError(f"Unexpected model input shape: {model.input_shape}")
        if model.output_shape != (None, len(class_labels)):
            raise ValueError(f"Unexpected model output shape: {model.output_shape}")
        return model, class_labels
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"The local model files are invalid: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not load {MODEL_PATH.name}: {exc}") from exc


def preprocess_image(image: Image.Image, image_size: int = 128) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size))
    array = tf.keras.utils.img_to_array(resized) / 255.0
    return np.expand_dims(array, axis=0)


def predict(
    model: tf.keras.Model,
    image: Image.Image,
    class_labels: tuple[str, ...],
) -> tuple[str, float, list[float]]:
    output = np.asarray(model.predict(preprocess_image(image), verbose=0))
    expected_shape = (1, len(class_labels))
    if output.shape != expected_shape:
        raise ValueError(
            f"Expected model output shape {expected_shape}, got {output.shape}."
        )
    scores = output[0]
    index = int(np.argmax(scores))
    label = class_labels[index]
    readable_label = f"Model output: {DISPLAY_LABELS[label]}"
    return readable_label, float(scores[index]), [float(score) for score in scores]


st.html(
    """
    <style>
    .hero-title, .panel-title {
        color: #172f3a;
        font-family: Georgia, "Times New Roman", serif;
        letter-spacing: -0.035em;
    }
    .brandline {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding-bottom: 1.1rem;
        border-bottom: 1px solid #d6dfdc;
        color: #173a43;
        font: 700 0.77rem/1 Arial, Helvetica, sans-serif;
        letter-spacing: 0.12em;
    }
    .brand-mark {
        display: inline-grid;
        width: 1.8rem;
        height: 1.8rem;
        place-items: center;
        border: 1px solid #397a72;
        border-radius: 50%;
        color: #286c65;
        font: 700 0.72rem/1 Arial, Helvetica, sans-serif;
        letter-spacing: 0;
    }
    .brand-subtitle {
        color: #73827f;
        font-weight: 400;
        letter-spacing: 0.04em;
    }
    .brand-status {
        margin-left: auto;
        color: #64736f;
        font-size: 0.64rem;
        letter-spacing: 0.14em;
    }
    .hero {
        padding: 2.8rem 0 2rem;
        border-bottom: 1px solid #d6dfdc;
    }
    .eyebrow, .section-label {
        color: #397a72;
        font: 700 0.68rem/1.4 Arial, Helvetica, sans-serif;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0.55rem 0 0.65rem;
        font-size: clamp(2.5rem, 5vw, 4.25rem);
        line-height: 1.05;
    }
    .hero-copy {
        max-width: 43rem;
        margin: 0;
        color: #566a6c;
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-label {
        margin: 0 0 0.8rem;
    }
    .side-panel {
        padding: 1.25rem 1.35rem;
        border: 1px solid #d6dfdc;
        border-radius: 0.35rem;
        background: #fbfcfa;
    }
    .panel-title {
        margin: 0 0 0.65rem;
        font-size: 1.35rem;
    }
    .side-panel p, .side-panel li {
        color: #566a6c;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .side-panel ul {
        margin: 0.55rem 0 0;
        padding-left: 1.1rem;
    }
    .disclaimer {
        padding-top: 1.1rem;
        border-top: 1px solid #d6dfdc;
        color: #697977;
        font-size: 0.78rem;
        line-height: 1.6;
    }
    </style>
    """
)

st.html(
    """
    <div class="brandline">
        <span class="brand-mark">N</span>
        <span>NEUROVISION <span class="brand-subtitle">/ MRI review</span></span>
        <span class="brand-status">EDUCATIONAL PROTOTYPE</span>
    </div>
    <div class="hero">
        <div class="eyebrow">Brain MRI classification</div>
        <h1 class="hero-title">A considered look<br>at every scan.</h1>
        <p class="hero-copy">
            Upload one MRI image to view the output of this MobileNetV2-based
            research model. This tool is intended for learning and demonstration,
            not for clinical decision-making.
        </p>
    </div>
    """
)

left_column, right_column = st.columns([1.35, 0.8], gap="large")

with left_column:
    st.html('<div class="section-label">01 / Image input</div>')
    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=("png", "jpg", "jpeg"),
        help="Supported formats: PNG and JPEG. DICOM files are not supported.",
    )

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            image.load()
            st.image(image, caption="Selected image", use_container_width=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("This file could not be read as a supported image.")

    run_prediction = st.button(
        "Run classification",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )

with right_column:
    st.html(
        """
        <div class="section-label">02 / What to expect</div>
        <div class="side-panel">
            <h3 class="panel-title">One image, four classes</h3>
            <p>The model returns a score for each class:</p>
            <ul>
                <li>Pituitary</li>
                <li>Glioma</li>
                <li>No tumor</li>
                <li>Meningioma</li>
            </ul>
            <p>Images are resized to 128 x 128 pixels and normalized before inference.</p>
        </div>
        """
    )

if run_prediction and image is not None:
    try:
        with st.spinner("Loading model and classifying the image..."):
            model, class_labels = load_model_cached()
            label, confidence, scores = predict(model, image, class_labels)

        st.markdown("---")
        st.html('<div class="section-label">03 / Model output</div>')
        result_column, scores_column = st.columns([1, 1.4], gap="large")
        with result_column:
            st.subheader(label)
            st.metric("Top class score", f"{confidence:.1%}")
            st.caption("Softmax score from the model; not a measure of clinical certainty.")
        with scores_column:
            chart_data = {
                "Model score": {
                    DISPLAY_LABELS[class_label]: score
                    for class_label, score in zip(class_labels, scores)
                }
            }
            st.bar_chart(chart_data)
        with st.expander("View raw model scores"):
            st.json(
                {
                    "class_order": list(class_labels),
                    "scores": scores,
                }
            )
    except RuntimeError as exc:
        st.error(str(exc))
    except (ValueError, tf.errors.InvalidArgumentError) as exc:
        st.error(f"The image could not be classified: {exc}")

st.html(
    """
    <div class="disclaimer">
        NeuroVision is a student research project. It is not a medical device,
        has not been validated for clinical use, and should not be used to
        diagnose or exclude a brain tumor.
    </div>
    """
)
