# Documentación Técnica

## Índice

1. [Decisiones de diseño](#decisiones-de-diseño)
2. [Arquitectura AWS](#arquitectura-aws)
3. [Pipeline de inferencia](#pipeline-de-inferencia)
4. [Despliegue paso a paso](#despliegue-paso-a-paso)
5. [Requisitos funcionales cubiertos](#requisitos-funcionales-cubiertos)
6. [Troubleshooting](#troubleshooting)

---

## Decisiones de diseño

### Por qué FastAPI + Mangum

FastAPI permite definir endpoints con validación automática via Pydantic y genera documentación OpenAPI sin configuración extra. Mangum adapta el handler ASGI de FastAPI para ejecutarse dentro de AWS Lambda sin cambios de código.

### Por qué Lambda contenedorizada

La imagen Docker permite incluir TensorFlow y OpenCV (dependencias pesadas) sin limitaciones del tamaño de capa Lambda. Facilita reproducibilidad entre entornos local y producción.

### Por qué CloudFront delante de S3

Provee HTTPS, compresión y caché del frontend estático sin necesidad de un servidor web. Las respuestas de error 403/404 se redirigen a `index.html` para soportar el enrutamiento SPA de React.

---

## Arquitectura AWS

```
┌─────────────────────────────────────────────────────────────┐
│                         Usuario                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTPS
                           ▼
              ┌────────────────────────┐
              │      CloudFront        │
              │  (CDN + HTTPS termina) │
              └──────┬─────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   ┌─────────────┐     ┌────────────────┐
   │  S3 Bucket  │     │  API Gateway   │
   │  (Frontend) │     │  HTTP API      │
   │  React SPA  │     └───────┬────────┘
   └─────────────┘             │
                               ▼
                    ┌─────────────────────┐
                    │   Lambda Function   │
                    │  (FastAPI + Mangum) │
                    │  Python 3.11 / ECR  │
                    └──────────┬──────────┘
                               │ (inicio en frío)
                               ▼
                    ┌─────────────────────┐
                    │   S3 Bucket Modelos │
                    │  mnist_cnn.h5       │
                    └─────────────────────┘
```

### Recursos Terraform creados

| Recurso | Nombre lógico | Propósito |
|---------|---------------|-----------|
| `aws_s3_bucket` | frontend | Archivos estáticos del SPA |
| `aws_s3_bucket` | modelos | Artefacto del modelo `.h5` |
| `aws_s3_bucket` | tf-state | Estado remoto de Terraform |
| `aws_cloudfront_distribution` | cdn | CDN + HTTPS del frontend |
| `aws_cloudfront_origin_access_control` | oac | Acceso seguro S3 → CloudFront |
| `aws_lambda_function` | api | Función con imagen ECR |
| `aws_apigatewayv2_api` | http-api | Entrada HTTP a Lambda |
| `aws_ecr_repository` | api | Registro de imagen Docker |
| `aws_iam_role` | lambda-exec | Permisos de ejecución Lambda |

---

## Pipeline de inferencia

El pipeline se ejecuta en `api/main.py` cada vez que se llama a `POST /predict` o `POST /predict/explain`. Las funciones clave son `decode_base64_image` y `preprocess_image`.

```
Imagen entrada (base64)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Fase 1 — Decodificación         decode_base64_image │
│                                                      │
│  base64.b64decode(string, validate=True)             │
│  → valida que sea base64 estrictamente bien formado  │
│  → Image.open() + img.verify() + Image.open()        │
│    (verify consume el stream; se necesita re-abrir)  │
│                                                      │
│  Errores que lanza (HTTP 400):                       │
│  · base64 mal formado                                │
│  · bytes decodificados vacíos                        │
│  · formato no reconocible (no es PNG/JPEG/BMP/etc.)  │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 2 — Escala de grises       preprocess_image    │
│                                                      │
│  img.convert("L")                                    │
│  → colapsa RGB/RGBA a un solo canal de luminancia    │
│  → fórmula PIL: L = 0.299R + 0.587G + 0.114B        │
│                                                      │
│  arr = np.array(img_gray, float32) / 255.0           │
│  → pasa el rango de [0, 255] a [0.0, 1.0]           │
│                                                      │
│  Guard: if arr.std() < 1e-6 → HTTP 400              │
│  → imagen completamente uniforme (blanco puro,       │
│    negro puro o sin contenido visual) es rechazada   │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 3 — Detección de fondo     preprocess_image    │
│                                                      │
│  border = concatenate(                               │
│    arr[0,:], arr[-1,:], arr[:,0], arr[:,-1]          │
│  )  → extrae los 4 bordes de la imagen               │
│                                                      │
│  if border.mean() > 0.5:                             │
│      arr = 1.0 - arr   ← inversión de colores        │
│                                                      │
│  Lógica: MNIST tiene fondo oscuro y trazo claro.     │
│  Si la media del borde supera 0.5 (borde claro),     │
│  se asume fondo blanco y se invierte para que        │
│  el dígito quede en blanco sobre negro.              │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 4 — Umbralización          preprocess_image    │
│                                                      │
│  threshold = max(0.2, min(0.7, arr.mean()))          │
│  → umbral dinámico anclado a la media de la imagen   │
│  → clampeo [0.2, 0.7] evita umbrales extremos que   │
│    borrarían el dígito o incluirían todo el ruido    │
│                                                      │
│  binary = (arr > threshold).astype(float32)          │
│  → binarización dura: 1.0 si supera el umbral,       │
│    0.0 si no                                         │
│  → elimina ruido de fondo y gradientes suaves,       │
│    dejando solo el trazo del dígito                  │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 5 — Resize 28×28           preprocess_image    │
│                                                      │
│  img_bin = Image.fromarray(binary * 255, mode="L")   │
│  img_resized = img_bin.resize((28,28), Image.LANCZOS)│
│                                                      │
│  LANCZOS (antes ANTIALIAS) aplica un filtro          │
│  sinc de alta calidad que preserva bordes del        │
│  trazo mejor que NEAREST o BILINEAR al reducir       │
│  imágenes de resolución arbitraria a 28×28.          │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 6 — Normalización final    preprocess_image    │
│                                                      │
│  out = np.array(img_resized, float32) / 255.0        │
│  → devuelve al rango [0.0, 1.0]                      │
│                                                      │
│  if out.max() > 0:                                   │
│      out = out / out.max()                           │
│  → garantiza que el valor máximo sea exactamente 1.0 │
│  → compensa atenuación introducida por LANCZOS       │
│                                                      │
│  out[np.newaxis, ..., np.newaxis]                    │
│  → reshape a tensor (1, 28, 28, 1)                   │
│    lote=1, alto=28, ancho=28, canal=1                │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 7 — Inferencia CNN         /predict            │
│                                                      │
│  probs = ml_model.predict(tensor, verbose=0)[0]      │
│  → salida softmax: vector de 10 probabilidades       │
│    que suman 1.0, una por cada dígito 0–9            │
│                                                      │
│  predicted_digit = np.argmax(probs)                  │
│  → índice con la probabilidad más alta               │
│                                                      │
│  confidence = np.max(probs)                          │
│  → probabilidad del dígito predicho                  │
│  → redondeado a 6 decimales en la respuesta          │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Fase 8 — Respuesta JSON                             │
│                                                      │
│  {                                                   │
│    "predicted_digit": 7,                             │
│    "confidence": 0.998341,                           │
│    "probabilities": {                                │
│      "0": 0.000001, "1": 0.000012, ..., "7": 0.998341│
│    }                                                 │
│  }                                                   │
└─────────────────────────────────────────────────────┘
```

### Grad-CAM — `/predict/explain` (opcional)

Cuando se llama a este endpoint, después de la Fase 7 se ejecuta `generar_gradcam`:

```
CNN predice clase C
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  1. Localizar última Conv2D                          │
│     _find_last_conv2d_layer_name(model)              │
│     → recorre model.layers en orden inverso          │
│                                                      │
│  2. Partir el modelo en dos                          │
│     feature_extractor → salida de la Conv2D          │
│     classifier_model  → capas Dense restantes        │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  3. GradientTape                                     │
│     tape.watch(conv_outputs)                         │
│     predictions = classifier_model(conv_outputs)     │
│     class_channel = predictions[:, clase_predicha]   │
│                                                      │
│     grads = tape.gradient(                           │
│         class_channel, conv_outputs                  │
│     )  → ∂score_clase / ∂activación_conv             │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  4. Mapa de calor                                    │
│     pooled = reduce_mean(grads, axis=(0,1,2))        │
│     → importancia global de cada filtro              │
│                                                      │
│     heatmap = sum(conv_outputs * pooled, axis=-1)    │
│     heatmap = relu(heatmap)   ← solo activaciones+  │
│     heatmap = heatmap / max(heatmap)  ← normalizar  │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  5. Superposición visual (OpenCV)                    │
│     cv2.resize(heatmap, (28,28), INTER_LINEAR)       │
│     cv2.applyColorMap(heatmap, COLORMAP_JET)         │
│     → escala de color azul (frío) → rojo (caliente)  │
│                                                      │
│     overlay = addWeighted(                           │
│         original_bgr, 0.6,                           │
│         heatmap_color, 0.4, 0.0                      │
│     )  → 60% imagen + 40% mapa de calor              │
│                                                      │
│     Retorna PNG en base64 → campo gradcam_base64     │
│     Si OpenCV no está disponible → None (silencioso) │
└─────────────────────────────────────────────────────┘
```

---

## Despliegue paso a paso

### Requisitos previos

```bash
# Verificar herramientas instaladas
aws --version        # AWS CLI v2
terraform --version  # >= 1.6
docker --version     # >= 24
```

Configurar credenciales AWS:
```bash
aws configure
# AWS Access Key ID: <key>
# AWS Secret Access Key: <secret>
# Default region name: us-east-1
```

### Paso 1 — Entrenar y subir el modelo

```bash
# Entrenar (desde el notebook) y luego subir a S3
aws s3 cp api/model/mnist_cnn.h5 \
  s3://<model-bucket-name>/mnist/mnist_cnn.h5
```

> El bucket de modelos se crea en el Paso 2. Si es el primer despliegue, hacer Paso 2 antes.

### Paso 2 — Infraestructura con Terraform

```bash
cd infra/lambda

# Copiar variables de ejemplo
cp ../../.env.example terraform.tfvars   # Ajustar valores

terraform init
terraform plan
terraform apply
```

Outputs importantes:
- `cloudfront_domain` — URL pública del frontend
- `api_gateway_url` — URL base del API

### Paso 3 — Build y push de imagen Docker

```bash
# Autenticarse en ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS \
  --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build y push
docker build -t kata-senior-ml-api api/
docker tag kata-senior-ml-api:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/kata-senior-ml-api:latest
docker push \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/kata-senior-ml-api:latest

# Actualizar Lambda con la nueva imagen
aws lambda update-function-code \
  --function-name kata-senior-ml-api \
  --image-uri <account-id>.dkr.ecr.us-east-1.amazonaws.com/kata-senior-ml-api:latest
```

### Paso 4 — Deploy del frontend

```bash
cd frontend
VITE_API_BASE_URL=https://<api-gateway-url> npm run build

aws s3 sync dist/ s3://<frontend-bucket-name>/ --delete

# Invalidar caché CloudFront
aws cloudfront create-invalidation \
  --distribution-id <distribution-id> \
  --paths "/*"
```

### CI/CD automático (GitHub Actions)

Configurar en el repositorio:
- **Secrets:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **Variables:** `AWS_REGION`

Cada push a `main` ejecuta los 4 jobs en secuencia: `validate → deploy_iac → deploy_api → deploy_frontend`.

---

## Requisitos funcionales cubiertos

| Requisito | Estado |
|-----------|--------|
| Sin transformers | ✅ |
| Sin modelos preentrenados ni fine-tuning | ✅ |
| Exportación de modelo permitida (`.h5`) | ✅ |
| Manejo de errores: base64 inválido | ✅ |
| Manejo de errores: imagen vacía | ✅ |
| Manejo de errores: formato inválido | ✅ |
| Manejo de errores: modelo no cargado | ✅ |
| API con documentación automática (Swagger) | ✅ |
| Despliegue serverless en AWS | ✅ |

---

## Troubleshooting

### Lambda retorna 500 al llamar `/predict`

Verificar que el modelo existe en S3 y que Lambda tiene la variable `MODEL_PATH` configurada correctamente. Revisar los logs en CloudWatch:
```bash
aws logs tail /aws/lambda/kata-senior-ml-api --follow
```

### `terraform apply` falla con "bucket already exists"

El workflow de CI crea el bucket de estado remotamente antes de `terraform init`. Si el bucket ya existe, ignorar el error del paso `aws s3api create-bucket`; Terraform usará el existente.

### Imagen no clasificada correctamente

El preprocesamiento espera dígitos oscuros sobre fondo claro (o viceversa). Si la imagen tiene bordes oscuros, el algoritmo asume fondo oscuro y la invierte automáticamente. Para depurar, llamar primero a `POST /preprocess` y verificar visualmente el resultado.

### CORS error en el frontend

Verificar que `VITE_API_BASE_URL` apunta exactamente al dominio del API Gateway (con `https://` y sin trailing slash). La API acepta cualquier origen en la configuración actual.
