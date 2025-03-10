terraform {
  required_providers {
    lambdalabs = {
      source = "elct9620/lambdalabs"
    }
  }
}

provider "lambdalabs" {
  api_key = "LAMBDALABS_API_KEY"
}

variable "instances" {
  default = ["cedana-demo-a", "cedana-demo-b"]
}

resource "lambdalabs_instance" "demo_instance" {
  for_each           = toset(var.instances)
  region_name        = "us-west-1"
  instance_type_name = "gpu_1x_a100"
  ssh_key_names      = ["terraform"]

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_ed25519")
    host        = self.ip
  }

  provisioner "remote-exec" {
    inline = [
      "git clone https://github.com/cedana/cedana-samples.git",
      "./cedana-samples/scripts/bootstrap-instance"
    ]
  }
}

output "instance_ips" {
  value = { for k, v in lambdalabs_instance.demo_instance : k => v.ip }
}
