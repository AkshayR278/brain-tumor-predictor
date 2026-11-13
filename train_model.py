from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_HANDLE = "masoudnickparvar/brain-tumor-mri-dataset"
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
SEED = 42
MAX_EPOCHS = 10


def _dataset(
    directory: Path,
    *,
    class_names: tuple[str, ...],
    subset: str | None = None,
    validation_split: float | None = None,
    shuffle: bool,
) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if validation_split is not None else None,
        validation_split=validation_split,
        subset=subset,
    )
    dataset = dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def _build_model() -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.025),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="mri_image")
    images = augmentation(inputs)
    images = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(images)
    features = backbone(images, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    features = tf.keras.layers.Dense(128, activation="relu")(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(features)

    model = tf.keras.Model(inputs, outputs, name="brain_mri_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _write_json_atomic(path: Path, content: dict[str, Any]) -> None:
    temporary_path = path.with_name(f"{path.stem}.pending{path.suffix}")
    temporary_path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def train_model(dataset_root: str | Path | None = None) -> dict[str, Any]:
    """Train and evaluate the educational MRI classifier, then export its files."""
    tf.keras.utils.set_random_seed(SEED)

    if dataset_root is None:
        dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    else:
        dataset_root = Path(dataset_root).expanduser().resolve()

    training_dir = dataset_root / "Training"
    testing_dir = dataset_root / "Testing"
    if not training_dir.is_dir() or not testing_dir.is_dir():
        raise FileNotFoundError(
            "Expected the dataset to contain Training/ and Testing/ folders "
            f"at {dataset_root}. Verify the Kaggle dataset download or pass "
            "the correct dataset root."
        )

    actual_classes = tuple(sorted(
        path.name for path in training_dir.iterdir() if path.is_dir()
    ))
    if actual_classes != CLASS_NAMES:
        raise ValueError(
            f"Expected class folders {CLASS_NAMES}, found {actual_classes}."
        )

    train_dataset = _dataset(
        training_dir,
        class_names=CLASS_NAMES,
        subset="training",
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
    )
    validation_dataset = _dataset(
        training_dir,
        class_names=CLASS_NAMES,
        subset="validation",
        validation_split=VALIDATION_SPLIT,
        shuffle=False,
    )
    test_dataset = _dataset(
        testing_dir,
        class_names=CLASS_NAMES,
        shuffle=False,
    )

    model = _build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    test_predictions = model.predict(test_dataset, verbose=0)
    predicted_labels = np.argmax(test_predictions, axis=1)
    true_labels = np.concatenate(
        [labels for _, labels in test_dataset.as_numpy_iterator()],
        axis=0,
    )
    test_loss = float(
        np.mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                true_labels,
                test_predictions,
            )
        )
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    results = {
        "dataset": DATASET_HANDLE,
        "class_names": list(CLASS_NAMES),
        "image_size": list(IMAGE_SIZE),
        "test_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "test_loss": float(test_loss),
        "confusion_matrix": confusion_matrix(
            true_labels,
            predicted_labels,
            labels=range(len(CLASS_NAMES)),
        ).tolist(),
        "classification_report": report,
        "epochs_completed": len(history.history["loss"]),
        "validation_split": VALIDATION_SPLIT,
        "seed": SEED,
    }

    model_path = PROJECT_DIR / "model.keras"
    labels_path = PROJECT_DIR / "model_labels.json"
    results_path = PROJECT_DIR / "evaluation.json"
    pending_model_path = PROJECT_DIR / "model.pending.keras"
    pending_labels_path = PROJECT_DIR / "model_labels.pending.json"

    model.save(pending_model_path)
    pending_labels_path.write_text(
        json.dumps({"class_names": list(CLASS_NAMES)}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(pending_model_path, model_path)
    os.replace(pending_labels_path, labels_path)
    _write_json_atomic(results_path, results)

    print(f"Saved trained model: {model_path.name}")
    print(f"Saved class order: {labels_path.name}")
    print(f"Saved held-out evaluation: {results_path.name}")
    print(f"Held-out test accuracy: {results['test_accuracy']:.4f}")
    return results


if __name__ == "__main__":
    train_model()
