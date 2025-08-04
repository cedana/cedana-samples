
output "cluster_endpoint" {
  description = "Endpoint for your EKS cluster."
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "The name of your EKS cluster."
  value       = module.eks.cluster_name
}

output "kubeconfig_command" {
  description = "Command to generate kubeconfig for the EKS cluster."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${var.cluster_name}"
}
