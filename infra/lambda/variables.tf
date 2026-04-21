variable "project_name" {
  description = "Nombre base del proyecto"
  type        = string
  default     = "kata-senior-ml"
}

variable "aws_region" {
  description = "Región de despliegue AWS"
  type        = string
  default     = "us-east-1"
}

variable "frontend_bucket_name" {
  description = "Bucket S3 para frontend estático"
  type        = string
}

variable "model_bucket_name" {
  description = "Bucket S3 para artefactos del modelo"
  type        = string
}

variable "lambda_image_uri" {
  description = "URI de la imagen Docker en ECR para Lambda"
  type        = string
}
