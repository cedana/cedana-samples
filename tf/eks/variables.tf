variable "aws_region" {
  description = "The AWS region to create resources in."
  type        = string
  default     = "us-east-1"
}

variable "k8s_version" {
  description = "The Kubernetes version for the EKS cluster. Ensure the version is supported by EKS and has a corresponding Ubuntu AMI."
  type        = string
  default     = "1.31"
}

variable "cluster_name" {
  description = "The name of the EKS cluster."
  type        = string
  default     = "cedana-ci"
}

variable "gpu_nodepool_name" {
  description = "The name of the GPU node pool."
  type        = string
  default     = "cedana-1xgpu-ci-pool"
}

variable "cpu_nodepool_name" {
  description = "The name of the CPU node pool."
  type        = string
  default     = "cedana-cpu-ci-pool"
}

variable "gpu_instance_type" {
  description = "The instance type for the GPU nodes."
  type        = string
  default     = "g4dn.xlarge"
}

variable "cpu_instance_type" {
  description = "The instance type for the CPU nodes."
  type        = string
  default     = "t3a.large"
}
