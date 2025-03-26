terraform {
  required_providers {
    lambdalabs = {
      source = "elct9620/lambdalabs"
    }
  }
}

provider "lambdalabs" {
  api_key = var.lambdalabs_api_key
}


resource "lambdalabs_instance" "demo_instance" {
  for_each           = toset(var.instance_names)
  region_name        = "us-west-1"
  instance_type_name = "gpu_1x_a100"
  ssh_key_names      = var.ssh_key_names

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_ed25519")
    host        = self.ip
  }

  provisioner "remote-exec" {
    inline = [
      "echo 'export CEDANA_URL=${var.cedana_url}' >> /etc/environment",
      "echo 'export CEDANA_AUTH_TOKEN=${var.cedana_auth_token}' >> /etc/environment",
      "echo 'export AWS_ACCESS_KEY_ID=${var.aws_access_key_id}' >> /etc/environment",
      "echo 'export AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}' >> /etc/environment",
      "echo 'export CEDANA_REMOTE=true' >> /etc/environment",
      "git clone https://github.com/cedana/cedana-samples.git",
      "./cedana-samples/scripts/bootstrap-instance"
    ]
  }
}

output "instance_ips" {
  value = { for k, v in lambdalabs_instance.demo_instance : k => v.ip }
}
