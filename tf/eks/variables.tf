variable "aws_region" {
  description = "The AWS region where the resources exist."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "The ID of the VPC where the EKS cluster will be deployed."
  type        = string
  default     = "vpc-09f07e664fdee09c8"
}

variable "subnet_ids" {
  description = "A list of subnet IDs where the EKS cluster and nodes will be deployed. Must be in the provided VPC."
  type        = list(string)
  default     = ["subnet-0363aec6f78777617", "subnet-0439575e73adb4515", "subnet-0b4eed876ce43949b"]
}

variable "k8s_version" {
  description = "The Kubernetes version for the EKS cluster. Ensure a Canonical Ubuntu 22.04 AMI exists for this version."
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
