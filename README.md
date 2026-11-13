# NeuroVision: Brain MRI Classification

An educational project for training and exploring a four-class brain MRI image
classifier. It uses TensorFlow/Keras, MobileNetV2 transfer learning, and a
Streamlit interface.

## Project files

```text
streamlit_app.py             Streamlit image review interface
train_model.py               Reproducible dataset, training, evaluation, and export pipeline
brain_tumour_detection.ipynb Notebook interface for the same training pipeline
model.keras                  Generated trained model (created by training)
model_labels.json            Model output class order (created by training)
evaluation.json              Held-out test metrics (created by training)
.streamlit/config.toml       Streamlit color theme
requirements.txt             Python dependencies
```

The training pipeline downloads the public
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
using KaggleHub. It expects `Training/` and `Testing/` directories with these
four class folders: `glioma`, `meningioma`, `notumor`, and `pituitary`. The
training code verifies the folders and uses that same alphabetical class order
for training, evaluation, export, and app inference.

## Setup on Windows

Use Python 3.10 or 3.11. Open PowerShell in the repository folder (the folder
containing `README.md` and `requirements.txt`):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

On the first training run, KaggleHub downloads the dataset and TensorFlow may
download the public ImageNet MobileNetV2 starting weights. Internet access is
required for those downloads; training then runs locally and can take several
minutes or longer depending on CPU/GPU and disk speed.

## Train and create the model files

**From the notebook:** open `brain_tumour_detection.ipynb` in VS Code, select
the `.venv` Python interpreter as its kernel, and run all cells. The notebook
uses the same tested pipeline as the script.

**From PowerShell:**

```powershell
.\.venv\Scripts\python.exe train_model.py
```

Training evaluates on the dataset's separate `Testing/` folder without
augmentation, then writes `model.keras`, `model_labels.json`, and
`evaluation.json` to the project directory.

The generated run in this project reached **88.31% accuracy** on 1,600 held-out
images (macro F1 **0.879**). Per-class recall was 69.0% for glioma, 86.3% for
meningioma, 99.5% for no tumor, and 98.5% for pituitary.

## Run the Streamlit app

After training has generated the model files:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Upload a PNG or JPEG MRI image and select **Run
classification**. The app displays the model's four class scores. It refuses to
run if the model files are missing, malformed, or incompatible, and reads the
class order from `model_labels.json`.

## Model and evaluation notes

- The notebook and `train_model.py` share one training implementation, so
  their preprocessing and exported model stay in sync.
- MobileNetV2 is initialized from ImageNet weights with its feature extractor
  frozen; a small classification head is trained on the MRI dataset.
- Training images are resized to 128 x 128 and normalized to [0, 1]. The model
  includes MobileNetV2's [-1, 1] input rescaling.
- Mild augmentation is applied only during training. Validation uses a
  deterministic split from `Training/`; final evaluation uses the separate
  `Testing/` images without augmentation.
- The displayed softmax values are model scores.
