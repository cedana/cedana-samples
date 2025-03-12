terraform {
  required_providers {
    crusoe = {
      source = "registry.terraform.io/crusoecloud/crusoe"
    }
  }
}

locals {
  my_ssh_key = file("~/.ssh/laptop-keypair.pub")
}

variable "instance_names" {
  type    = list(string)
  default = ["cedana-demo-a", "cedana-demo-b"]
}


resource "crusoe_compute_instance" "cedana_demo" {
  for_each = toset(var.instance_names)

  name     = each.value
  type     = "a40.1x"
  location = "us-northcentral1-a"

  image = "ubuntu22.04:latest"

  ssh_key        = local.my_ssh_key
  startup_script = <<-EOF
  #!/bin/bash
  set -e  # Exit on error

  echo "export CEDANA_URL='${var.cedana_url}'" >> /etc/environment
  echo "export CEDANA_AUTH_TOKEN='${var.cedana_auth_token}'" >> /etc/environment
  echo "export AWS_ACCESS_KEY_ID='${var.aws_access_key_id}'" >> /etc/environment
  echo "export AWS_SECRET_ACCESS_KEY='${var.aws_secret_access_key}'" >> /etc/environment

  # Ensure environment variables are loaded
  source /etc/environment

  # Retry logic for git clone (up to 5 attempts with exponential backoff)
  for i in {1..5}; do
    git clone --depth 1 https://github.com/cedana/cedana-samples.git && break
    echo "Git clone failed, retrying in $((2**i)) seconds..."
    sleep $((2**i))
  done

  # Ensure bootstrap script is executable
  chmod +x cedana-samples/scripts/bootstrap-instance

  # Run bootstrap script with logging
  ./cedana-samples/scripts/bootstrap-instance > /var/log/bootstrap.log 2>&1
EOF

}

output "instance_ips" {
  value = { for k, v in crusoe_compute_instance.cedana_demo : k => v.network_interfaces[0].public_ipv4.address }
}
