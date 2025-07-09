output "cluster_endpoint" {
  description = "Endpoint for your EKS cluster."
  value       = aws_eks_cluster.cedana_ci_cluster.endpoint
}

output "cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with your cluster."
  value       = aws_eks_cluster.cedana_ci_cluster.certificate_authority[0].data
}

output "cluster_name" {
  description = "The name of your EKS cluster."
  value       = aws_eks_cluster.cedana_ci_cluster.name
}

output "kubeconfig_command" {
  description = "Command to generate kubeconfig for the EKS cluster."
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${var.cluster_name}"
}
