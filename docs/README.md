# Documentacion tecnica

## Objetivo
Implementar una solucion de clasificacion de digitos manuscritos (MNIST) con:
- Entrenamiento en notebook con Keras.
- Exportacion de modelo `.h5`.
- API FastAPI con endpoint `POST /predict`.
- Opcion de despliegue serverless en AWS (Lambda + API Gateway).

## Estructura del repositorio
- `notebook/`: entrenamiento y experimentacion.
- `api/`: servicio de inferencia.
- `infra/lambda/`: infraestructura como codigo con Terraform.
- `scripts/`: utilidades de prueba y soporte.

## Requisitos funcionales cubiertos
- Sin transformers.
- Sin modelos preentrenados ni fine-tuning.
- Exportacion de modelo permitida (`.h5`).
- API con manejo de errores:
  - base64 invalido
  - imagen vacia
  - formato de imagen no valido

## Flujo sugerido
1. Entrenar en `notebook/mnist_classifier.ipynb`.
2. Guardar artefacto en `api/model/mnist_cnn.h5`.
3. Levantar API localmente.
4. Validar con `scripts/test_api.py`.
5. Desplegar en AWS usando `infra/lambda/`.
