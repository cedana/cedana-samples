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

variable "ssh_key_names" {
  description = "List of SSH key names"
  type        = list(string)
  default     = ["laptop-keypair"]
}

variable "instance_names" {
  description = "List of instance names"
  type        = list(string)
  default     = ["cedana-demo-a", "cedana-demo-b"]
}


variable "lambdalabs_api_key" {
  description = "Lambda Labs API key"
  type        = string
  sensitive   = true
}
