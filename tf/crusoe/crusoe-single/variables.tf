variable "cedana_url" {
  description = "Cedana API URL"
  type        = string
}

variable "cedana_auth_token" {
  description = "Cedana authentication token"
  type        = string
  sensitive   = true
}

variable "aws_access_key_id" {
  description = "AWS Access Key ID"
  type        = string
  sensitive   = true
}

variable "aws_secret_access_key" {
  description = "AWS Secret Access Key"
  type        = string
  sensitive   = true
}

variable "ssh_key" {
  description = "Path to the SSH public key"
  type        = string
  default     = "~/.ssh/laptop-keypair.pub"
}

variable "instance_names" {
  description = "List of instance names"
  type        = list(string)
  default     = ["cedana-demo-a", "cedana-demo-b"]
}

variable "instance_type" {
  description = "Instance type"
  type        = string
  default     = "a40.1x"
}
