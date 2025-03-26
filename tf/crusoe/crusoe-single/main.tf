terraform {
  required_providers {
    crusoe = {
      source = "registry.terraform.io/crusoecloud/crusoe"
    }
  }
}


resource "crusoe_compute_instance" "cedana_demo" {
  for_each = toset(var.instance_names)

  name     = each.value
  type     = var.instance_type
  location = "us-northcentral1-a"

  image   = "ubuntu22.04:latest"
  ssh_key = file(var.ssh_key)

  startup_script = <<-EOF
  #!/bin/bash
  set -e  # Exit on error

  # Write environment variables to /etc/environment
  echo "CEDANA_URL='${var.cedana_url}'" >> /etc/environment
  echo "CEDANA_AUTH_TOKEN='${var.cedana_auth_token}'" >> /etc/environment
  echo "AWS_ACCESS_KEY_ID='${var.aws_access_key_id}'" >> /etc/environment
  echo "AWS_SECRET_ACCESS_KEY='${var.aws_secret_access_key}'" >> /etc/environment
  echo "CEDANA_REMOTE=true" >> /etc/environment

  # Reload environment variables
  export CEDANA_URL="${var.cedana_url}"
  export CEDANA_AUTH_TOKEN="${var.cedana_auth_token}"
  export AWS_ACCESS_KEY_ID="${var.aws_access_key_id}"
  export AWS_SECRET_ACCESS_KEY="${var.aws_secret_access_key}"
  export CEDANA_REMOTE=true

  # Retry logic for git clone (up to 5 attempts with exponential backoff)
  for i in {1..5}; do
    git clone --depth 1 https://github.com/cedana/cedana-samples.git /root/cedana-samples && break
    echo "Git clone failed, retrying in $((2**i)) seconds..."
    sleep $((2**i))
  done

  EOF
}

output "instance_ips" {
  value = { for k, v in crusoe_compute_instance.cedana_demo : k => v.network_interfaces[0].public_ipv4.address }
}
