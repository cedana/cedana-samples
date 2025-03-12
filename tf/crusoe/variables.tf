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
