# kata-senior-ml

Kata tecnica de Machine Learning Senior (clasificacion de digitos MNIST).

## Estructura oficial (sin redundancias)
```
kata-senior-ml/
├── notebook/
│   └── mnist_classifier.ipynb
├── api/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── model/
│       ├── .gitkeep
│       └── mnist_cnn.h5
├── infra/
│   └── lambda/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── docs/
│   └── README.md
├── scripts/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── deploy.yml
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

## Requisitos
- Python 3.11 recomendado.
- TensorFlow (sin transformers, sin preentrenados).

## Setup rapido
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api/requirements.txt
```

## Entrenamiento y exportacion
1. Abre `notebook/mnist_classifier.ipynb`.
2. Ejecuta todas las celdas.
3. El modelo se exporta en:
   - `api/model/mnist_cnn.h5`

## Ejecutar API
```bash
python -m uvicorn api.main:app --reload
```

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Frontend sencillo (demo)
Con la API encendida en una terminal, en otra terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Abre:
- `http://127.0.0.1:5173`

Variable de entorno del frontend:
- `VITE_API_BASE_URL` (por defecto `http://127.0.0.1:8000`)

La pagina permite:
- subir imagen
- cargar ejemplo real de MNIST para demo controlada
- tomar foto desde camara del dispositivo
- llamar `POST /preprocess`
- llamar `POST /predict` (si usas imagen original)
- llamar `POST /predict-preprocessed` automaticamente cuando usas la salida de preprocess
- guardar imagen etiquetada (`0-9`) con `POST /save-labeled-image`

Las imagenes guardadas quedan en:
- `data/labeled/<etiqueta>/archivo.png`

Endpoint util para demo:
- `GET /sample-mnist?index=123` devuelve imagen base64 + etiqueta real.

Preprocesamiento usado:
- `simple`: invertir segun fondo + umbral + resize 28x28

## Probar prediccion
```bash
python scripts/test_api.py --url http://127.0.0.1:8000/predict --index 0
```

## Preprocesar primero y luego predecir
1. Llama `POST /preprocess` con la imagen base64.
2. Toma `processed_image` de la respuesta.
3. Envia ese `processed_image` a `POST /predict`.

## Manejo de errores de la API
`POST /preprocess` y `POST /predict` contemplan:
- base64 invalido
- imagen vacia
- formato no valido

## Deploy en AWS (IaC -> API -> Frontend)
Hay un solo pipeline en `.github/workflows/deploy.yml` y corre en este orden:

1. `deploy_iac`
2. `deploy_api`
3. `deploy_frontend`

Con esto se evita que frontend y backend suban antes de que la infraestructura exista.

### 1) Credenciales para GitHub Actions
Se usa una sola identidad AWS para el pipeline completo.  
Configura estos `Secrets` en GitHub:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Y esta `Variable`:

- `AWS_REGION` (ejemplo: `us-east-1`)

### 2) Qué crea IaC
El job `deploy_iac` crea/actualiza:

- bucket S3 del frontend (website hosting)
- bucket S3 de modelos
- rol de ejecucion para Lambda
- Lambda basada en imagen Docker
- API Gateway HTTP API (ruta default a Lambda)

Tambien asegura:

- repositorio ECR para la imagen del API
- bucket S3 para estado remoto de Terraform

### 3) Modelo obligatorio para desplegar API
Antes de desplegar API, sube el modelo:

```bash
aws s3 cp <ruta-local-del-modelo.h5> s3://kata-senior-ml-models-<aws-account-id>-<aws-region>/mnist/mnist_cnn.h5
```

El job `deploy_api` falla si ese archivo no existe.
