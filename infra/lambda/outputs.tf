output "api_url" {
  description = "URL base del API Gateway"
  value       = aws_apigatewayv2_stage.prod.invoke_url
}

output "lambda_function_name" {
  description = "Nombre de la función Lambda desplegada"
  value       = aws_lambda_function.mnist_api.function_name
}

output "frontend_bucket_name" {
  description = "Nombre del bucket S3 para frontend"
  value       = aws_s3_bucket.artifacts.id
}

output "frontend_website_url" {
  description = "URL del website hosting de S3 para frontend"
  value       = aws_s3_bucket_website_configuration.frontend_website.website_endpoint
}

output "model_bucket_name" {
  description = "Nombre del bucket S3 para artefactos de modelo"
  value       = aws_s3_bucket.model.id
}
