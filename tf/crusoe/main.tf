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
    echo "export CEDANA_URL='${var.cedana_url}'" >> /etc/environment
    echo "export CEDANA_AUTH_TOKEN='${var.cedana_auth_token}'" >> /etc/environment
    echo "export AWS_ACCESS_KEY_ID='${var.aws_access_key_id}'" >> /etc/environment
    echo "export AWS_SECRET_ACCESS_KEY='${var.aws_secret_access_key}'" >> /etc/environment
    git clone https://github.com/cedana/cedana-samples.git
    ./cedana-samples/scripts/bootstrap-instance
  EOF
}

output "instance_ips" {
  value = { for k, v in crusoe_compute_instance.cedana_demo : k => v.network_interfaces[0].public_ipv4.address }
}
