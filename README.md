# kata-senior-ml

Kata técnica de Machine Learning Senior para clasificación de dígitos MNIST.

## Stack actual
- **API:** FastAPI + TensorFlow (`api/main.py`)
- **Frontend:** React + Vite (`frontend/src/App.jsx`)
- **Infraestructura:** Terraform para Lambda + API Gateway + S3 + CloudFront (`infra/lambda`)
- **CI/CD:** GitHub Actions con flujo secuencial IaC -> API -> Frontend (`.github/workflows/deploy.yml`)

## Estructura principal
```text
kata-senior-ml/
├── api/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── Dockerfile.bootstrap
│   ├── bootstrap_handler.py
│   └── model/
│       └── .gitkeep
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
├── .github/
│   └── workflows/
│       └── deploy.yml
├── requirements.txt
└── README.md
```

## Requisitos
- Python 3.11
- Node.js 18+
- npm 9+
- Terraform 1.6+

## Setup local
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r api/requirements.txt
```

## Entrenamiento y modelo
1. Abre `notebook/mnist_classifier.ipynb`.
2. Ejecuta todas las celdas.
3. Exporta el modelo a `api/model/mnist_cnn.h5` para ejecución local.

Nota: el archivo `.h5` no se versiona en Git.

## Ejecutar API local
```bash
python -m uvicorn api.main:app --reload
```

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Ejecutar frontend local (React + Vite)
```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Abre `http://127.0.0.1:5173`.

Variable de entorno usada por frontend:
- `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`)

## Funcionalidad de la app
- Cargar muestra aleatoria de MNIST (`GET /sample-mnist`)
- Cargar imagen local
- Capturar foto con cámara
- Preprocesar imagen (`POST /preprocess`)
- Inferir desde imagen original (`POST /predict`)
- Inferir desde imagen preprocesada (`POST /predict-preprocessed`)
- Guardar imágenes locales etiquetadas desde el frontend

## Prueba rápida de inferencia por script
```bash
python scripts/test_api.py --url http://127.0.0.1:8000/predict --index 0
```

## Deploy en AWS
El workflow `deploy.yml` ejecuta los jobs en secuencia:
1. `validate`
2. `deploy_iac`
3. `deploy_api`
4. `deploy_frontend`

Esto asegura que frontend y API no se despliegan antes de tener infraestructura.

### Secrets y variables en GitHub
**Secrets**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

**Variable**
- `AWS_REGION` (ejemplo: `us-east-1`)

### Qué crea `deploy_iac`
- Bucket S3 privado para frontend
- Distribución CloudFront HTTPS para frontend (sin dominio custom)
- Bucket S3 para modelos
- Repositorio ECR del API
- Bucket S3 para Terraform state remoto
- IAM Role de ejecución para Lambda
- Lambda con imagen bootstrap
- API Gateway HTTP API integrado a Lambda

### Requisito para `deploy_api`
Antes del deploy, el modelo debe existir en el bucket de modelos creado por IaC:
```bash
s3://<nombre-del-bucket-modelos>/mnist/mnist_cnn.h5
```

Comando de referencia (genérico):
```bash
aws s3 cp <ruta-local-del-modelo.h5> s3://<nombre-del-bucket-modelos>/mnist/mnist_cnn.h5
```
