# Kata Senior ML — Clasificador de Dígitos MNIST

Aplicación end-to-end de Machine Learning para clasificación de dígitos manuscritos (0–9) usando una CNN entrenada sobre MNIST, expuesta como API serverless en AWS.

## Stack


| Capa                                                                                              | Tecnología                                      |
| ------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| Modelo                                                                                            | TensorFlow/Keras — CNN entrenada sobre MNIST    |
| API                                                                                               | Python 3.11, FastAPI, Mangum (adaptador Lambda) |
| Frontend                                                                                          | React 18.3, Vite 5.4                            |
| Infraestructura                                                                                   | AWS Lambda + API Gateway + S3 + CloudFront      |
| IaC[https://github.com/Juancho2406/kata-senior-ml](https://github.com/Juancho2406/kata-senior-ml) | Terraform 1.6+                                  |
| CI/CD                                                                                             | GitHub Actions                                  |


## Arquitectura

```
Usuario
  │
  ▼
CloudFront (HTTPS)
  ├── /            → S3 (React app)
  └── /api/*       → API Gateway → Lambda (FastAPI)
                                        │
                                        └── S3 (modelo mnist_cnn.h5)
```

## Funcionalidades

- Clasificar dígitos desde imagen cargada, cámara o muestra MNIST
- Preprocesamiento automático (escala de grises, umbral, resize 28×28)
- Visualización Grad-CAM de las zonas de atención del modelo
- Evaluación por lotes de 32–128 imágenes del dataset de prueba
- Guardar imágenes etiquetadas localmente para reentrenamiento

## Estructura del proyecto

```
kata-senior-ml/
├── api/
│   ├── main.py               # FastAPI app + endpoints
│   ├── requirements.txt
│   ├── Dockerfile            # Imagen de producción para Lambda
│   ├── Dockerfile.bootstrap  # Imagen mínima para bootstrap inicial
│   ├── bootstrap_handler.py  # Handler placeholder para primera provisión
│   └── model/
│       └── .gitkeep          # El archivo .h5 NO se versiona en Git
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
├── infra/
│   └── lambda/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── notebook/
│   └── mnist_classifier.ipynb
├── scripts/
│   └── test_api.py
├── docs/
│   └── README.md             # Documentación técnica detallada
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .env.example              # Variables de entorno de referencia
└── requirements.txt
```

## Requisitos previos

- Python 3.11
- Node.js 18+ / npm 9+
- Terraform 1.6+
- AWS CLI configurado (para despliegue)
- Docker (para build de imagen Lambda)

## Setup local

### 1. Entorno Python

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### 2. Entrenar el modelo

```bash
# Abrir y ejecutar todas las celdas del notebook
jupyter notebook notebook/mnist_classifier.ipynb
# El modelo se guarda en api/model/mnist_cnn.h5
```

### 3. Ejecutar la API

```bash
python -m uvicorn api.main:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 4. Ejecutar el frontend

```bash
cd frontend
cp .env.example .env        # Ajustar VITE_API_BASE_URL si es necesario
npm install
npm run dev
```

Abrir `http://127.0.0.1:5173`

## Variables de entorno

### API / Lambda


| Variable     | Descripción                | Default              |
| ------------ | -------------------------- | -------------------- |
| `MODEL_PATH` | Ruta local al modelo `.h5` | `model/mnist_cnn.h5` |


### Frontend


| Variable            | Descripción      | Default                 |
| ------------------- | ---------------- | ----------------------- |
| `VITE_API_BASE_URL` | URL base del API | `http://127.0.0.1:8000` |


### CI/CD (GitHub Secrets y Variables)


| Nombre                  | Tipo     | Descripción                  |
| ----------------------- | -------- | ---------------------------- |
| `AWS_ACCESS_KEY_ID`     | Secret   | Credenciales de despliegue   |
| `AWS_SECRET_ACCESS_KEY` | Secret   | Credenciales de despliegue   |
| `AWS_REGION`            | Variable | Región AWS (ej. `us-east-1`) |


Ver `.env.example` para las variables de Terraform.

## Prueba rápida del API

```bash
python scripts/test_api.py --url http://127.0.0.1:8000/predict --index 0
```

## Endpoints principales


| Método | Ruta                    | Descripción                           |
| ------ | ----------------------- | ------------------------------------- |
| `GET`  | `/health`               | Estado del servicio y modelo          |
| `POST` | `/predict`              | Clasificar imagen (base64)            |
| `POST` | `/predict-preprocessed` | Clasificar imagen 28×28 preprocesada  |
| `POST` | `/predict/explain`      | Clasificar con mapa Grad-CAM          |
| `GET`  | `/sample-mnist`         | Obtener muestra aleatoria del dataset |
| `POST` | `/sample-mnist-batch`   | Evaluar lote de 32–128 imágenes       |
| `POST` | `/preprocess`           | Preprocesar imagen a formato MNIST    |
| `POST` | `/save-labeled-image`   | Guardar imagen etiquetada localmente  |


## Despliegue en AWS

Ver `[docs/README.md](docs/README.md)` para el flujo completo de despliegue, incluyendo requisitos previos y pasos manuales.

El pipeline CI/CD (`deploy.yml`) ejecuta automáticamente en push a `main`:

1. `validate` — Verifica dependencias Python
2. `deploy_iac` — Aplica Terraform (crea/actualiza infra AWS)
3. `deploy_api` — Construye imagen Docker, sube a ECR, actualiza Lambda
4. `deploy_frontend` — Build Vite, sync a S3, invalida caché CloudFront

## Modelo

### Resultados


| Métrica                  | Valor                     |
| ------------------------ | ------------------------- |
| **Test accuracy**        | **99.56%**                |
| Test loss                | 0.0155                    |
| Parámetros totales       | 871,018 (~3.32 MB)        |
| Tamaño del archivo `.h5` | ~10 MB                    |
| Framework                | TensorFlow 2.16.2 / Keras |


### Arquitectura — `mnist_cnn`

```
Input (28, 28, 1)
│
├── Conv2D(32, 3×3, relu, padding=same)   →  (28, 28, 32)   │ 320 params
├── BatchNormalization                                         │ 128 params
├── Conv2D(32, 3×3, relu, padding=same)   →  (28, 28, 32)   │ 9,248 params
├── MaxPooling2D(2×2)                      →  (14, 14, 32)
├── Dropout(0.25)
│
├── Conv2D(64, 3×3, relu, padding=same)   →  (14, 14, 64)   │ 18,496 params
├── BatchNormalization                                         │ 256 params
├── Conv2D(64, 3×3, relu, padding=same)   →  (14, 14, 64)   │ 36,928 params
├── MaxPooling2D(2×2)                      →  (7, 7, 64)
├── Dropout(0.25)
│
├── Flatten                                →  (3136,)
├── Dense(256, relu)                       →  (256,)          │ 803,072 params
├── Dropout(0.5)
└── Dense(10, softmax)                     →  (10,)           │ 2,570 params
```

### Entrenamiento


| Hiperparámetro | Valor                         |
| -------------- | ----------------------------- |
| Optimizador    | Adam (lr=1e-3)                |
| Loss           | Categorical Crossentropy      |
| Batch size     | 128                           |
| Épocas máximas | 20 (EarlyStopping patience=5) |
| Split          | 54K train / 6K val / 10K test |


**Data augmentation aplicado:**

- Rotación ±10°, zoom 10%, desplazamiento horizontal/vertical 10%, shear 10%
- Inversión de colores del dataset completo (duplica ejemplos a 108K) — el modelo aprende tanto dígitos blancos sobre negro (MNIST estándar) como negros sobre blanco (fotografías reales)

### Restricciones de la kata

- Sin transformers
- Sin modelos preentrenados ni fine-tuning
- Modelo entrenado desde cero exclusivamente sobre MNIST

**Nota:** El archivo `.h5` no se versiona en Git. Debe subirse manualmente a S3 antes del primer despliegue del API (ver `[docs/README.md](docs/README.md)`).