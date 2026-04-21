"""
API FastAPI — Clasificación de dígitos MNIST
Kata Técnica Senior ML
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, field_validator
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "api/model/mnist_cnn.h5")
TARGET_SIZE = (28, 28)
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# Estado global del modelo
# ---------------------------------------------------------------------------
ml_model: keras.Model | None = None


# ---------------------------------------------------------------------------
# Lifespan (carga/descarga del modelo)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global ml_model
    logger.info("Cargando modelo desde %s ...", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modelo no encontrado en '{MODEL_PATH}'. "
            "Ejecuta el notebook de entrenamiento primero."
        )
    ml_model = keras.models.load_model(MODEL_PATH)
    logger.info("Modelo cargado correctamente. Arquitectura: %s", ml_model.name)
    yield
    logger.info("Cerrando la aplicación.")
    ml_model = None


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MNIST Digit Classifier",
    description=(
        "Clasificación de dígitos manuscritos (0-9) con una CNN entrenada "
        "desde cero sobre el dataset MNIST."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # Imagen codificada en base64

    @field_validator("image")
    @classmethod
    def image_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("El campo 'image' no puede estar vacío.")
        return value.strip()


class PredictResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: Dict[str, float]


class PreprocessResponse(BaseModel):
    processed_image: str
    width: int
    height: int
    pixel_min: float
    pixel_max: float


class SaveLabeledImageRequest(BaseModel):
    image: str
    label: int
    use_preprocess: bool = False

    @field_validator("image")
    @classmethod
    def image_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("El campo 'image' no puede estar vacío.")
        return value.strip()

    @field_validator("label")
    @classmethod
    def label_must_be_digit(cls, value: int) -> int:
        if value < 0 or value > 9:
            raise ValueError("El campo 'label' debe estar entre 0 y 9.")
        return value


class SaveLabeledImageResponse(BaseModel):
    message: str
    saved_path: str
    label: int
    saved_preprocessed: bool


class MnistSampleResponse(BaseModel):
    image: str
    label: int
    index: int


# ---------------------------------------------------------------------------
# Utilidades de preprocesamiento
# ---------------------------------------------------------------------------
def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Decodifica un string base64 a un objeto PIL Image.

    Raises:
        HTTPException 400: base64 inválido o bytes no son una imagen válida.
    """
    try:
        img_bytes = base64.b64decode(b64_string, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="base64 inválido: el string no es un base64 bien formado.",
        ) from exc

    if len(img_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Imagen vacía: los bytes decodificados están vacíos.",
        )

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        img = Image.open(io.BytesIO(img_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Formato no válido: los bytes no corresponden a una imagen "
                "reconocible (PNG, JPEG, BMP, etc.)."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No se pudo abrir la imagen: {exc}",
        ) from exc

    return img


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Pipeline simple: gris -> invertir segun fondo -> umbral -> 28x28.
    """
    img_gray = img.convert("L")
    arr = np.array(img_gray, dtype="float32") / 255.0

    if arr.std() < 1e-6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Imagen vacía o uniforme: no contiene información visual útil.",
        )

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(border.mean()) > 0.5:
        arr = 1.0 - arr

    threshold = max(0.2, min(0.7, float(arr.mean())))
    binary = (arr > threshold).astype("float32")

    img_bin = Image.fromarray((binary * 255.0).astype("uint8"), mode="L")
    img_resized = img_bin.resize(TARGET_SIZE, Image.LANCZOS)
    out = np.array(img_resized, dtype="float32") / 255.0

    if float(out.max()) > 0:
        out = out / float(out.max())

    return out[np.newaxis, ..., np.newaxis]


def tensor_to_base64_image(tensor: np.ndarray) -> str:
    """
    Convierte tensor (1, 28, 28, 1) normalizado en [0, 1] a PNG base64.
    """
    arr = np.squeeze(tensor, axis=(0, 3))
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype("uint8")
    pil_image = Image.fromarray(arr_uint8, mode="L")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocessed_base64_to_tensor(img: Image.Image) -> np.ndarray:
    """
    Convierte una imagen ya preprocesada (esperada en estilo MNIST)
    a tensor (1, 28, 28, 1) sin aplicar el pipeline pesado otra vez.
    """
    img_gray = img.convert("L").resize(TARGET_SIZE, Image.LANCZOS)
    arr = np.array(img_gray, dtype="float32") / 255.0
    if arr.std() < 1e-6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Imagen preprocesada vacía o uniforme.",
        )
    return arr[np.newaxis, ..., np.newaxis]


def mnist_image_to_base64(arr_2d: np.ndarray) -> str:
    img = Image.fromarray(arr_2d.astype("uint8"), mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": ml_model is not None}


@app.get("/health", tags=["Health"])
def health():
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está cargado.",
        )
    return {
        "status": "healthy",
        "model": ml_model.name,
        "input_shape": str(ml_model.input_shape),
    }


@app.post(
    "/preprocess",
    response_model=PreprocessResponse,
    status_code=status.HTTP_200_OK,
    tags=["Preprocessing"],
    summary="Preprocesa una imagen para MNIST (28x28)",
)
def preprocess(request: PredictRequest):
    img = decode_base64_image(request.image)
    tensor = preprocess_image(img)
    processed_b64 = tensor_to_base64_image(tensor)
    arr = np.squeeze(tensor, axis=(0, 3))

    return PreprocessResponse(
        processed_image=processed_b64,
        width=TARGET_SIZE[0],
        height=TARGET_SIZE[1],
        pixel_min=float(arr.min()),
        pixel_max=float(arr.max()),
    )


@app.post(
    "/save-labeled-image",
    response_model=SaveLabeledImageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Dataset"],
    summary="Guarda una imagen etiquetada localmente para reentrenamiento",
)
def save_labeled_image(request: SaveLabeledImageRequest):
    img = decode_base64_image(request.image)
    label_dir = os.path.join("data", "labeled", str(request.label))
    os.makedirs(label_dir, exist_ok=True)

    if request.use_preprocess:
        tensor = preprocess_image(img)
        arr = np.squeeze(tensor, axis=(0, 3))
        image_to_save = Image.fromarray(
            np.clip(arr * 255.0, 0, 255).astype("uint8"),
            mode="L",
        )
    else:
        image_to_save = img.convert("L")

    filename = (
        f"label_{request.label}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    )
    save_path = os.path.join(label_dir, filename)
    image_to_save.save(save_path, format="PNG")

    return SaveLabeledImageResponse(
        message="Imagen guardada correctamente.",
        saved_path=save_path,
        label=request.label,
        saved_preprocessed=request.use_preprocess,
    )


@app.get(
    "/sample-mnist",
    response_model=MnistSampleResponse,
    status_code=status.HTTP_200_OK,
    tags=["Dataset"],
    summary="Obtiene una imagen de ejemplo del test set MNIST",
)
def sample_mnist(index: int = 0):
    (_, _), (x_test, y_test) = mnist.load_data()
    if index < 0 or index >= len(x_test):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"index fuera de rango. Debe estar entre 0 y {len(x_test)-1}.",
        )
    img_b64 = mnist_image_to_base64(x_test[index])
    return MnistSampleResponse(
        image=img_b64,
        label=int(y_test[index]),
        index=index,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Clasifica un dígito manuscrito",
)
def predict(request: PredictRequest):
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible.",
        )

    img = decode_base64_image(request.image)
    tensor = preprocess_image(img)

    probs: np.ndarray = ml_model.predict(tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    logger.info("Predicción: %d (confianza: %.4f)", predicted_digit, confidence)

    return PredictResponse(
        predicted_digit=predicted_digit,
        confidence=round(confidence, 6),
        probabilities={str(i): round(float(probs[i]), 6) for i in range(NUM_CLASSES)},
    )


@app.post(
    "/predict-preprocessed",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Clasifica una imagen ya preprocesada (28x28)",
)
def predict_preprocessed(request: PredictRequest):
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo no está disponible.",
        )

    img = decode_base64_image(request.image)
    tensor = preprocessed_base64_to_tensor(img)

    probs: np.ndarray = ml_model.predict(tensor, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    logger.info(
        "Predicción preprocesada: %d (confianza: %.4f)", predicted_digit, confidence
    )

    return PredictResponse(
        predicted_digit=predicted_digit,
        confidence=round(confidence, 6),
        probabilities={str(i): round(float(probs[i]), 6) for i in range(NUM_CLASSES)},
    )


# ---------------------------------------------------------------------------
# Manejadores de error globales
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    del request
    logger.error("Error inesperado: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor."},
    )


# AWS Lambda entrypoint for API Gateway HTTP API events
handler = Mangum(app)
