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

output "frontend_cloudfront_domain" {
  description = "Dominio HTTPS de CloudFront para el frontend"
  value       = aws_cloudfront_distribution.frontend_cdn.domain_name
}

output "frontend_cloudfront_distribution_id" {
  description = "ID de la distribucion CloudFront del frontend"
  value       = aws_cloudfront_distribution.frontend_cdn.id
}

output "model_bucket_name" {
  description = "Nombre del bucket S3 para artefactos de modelo"
  value       = aws_s3_bucket.model.id
}
