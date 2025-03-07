terraform {
  required_providers {
    crusoe = {
      source = "registry.terraform.io/crusoecloud/crusoe"
    }
  }
}

locals {
  my_ssh_key = file("~/.ssh/id_ed25519.pub")
}

resource "crusoe_compute_instance" "cedana-demo-b" {
  name     = "cedana-demo-a"
  type     = "a10.1x"
  location = "us-northcentral1-a"

  # specify the base image
  image = "ubuntu22.04:latest"

  disks = [
    // disk attached at startup
    {
      id              = crusoe_storage_disk.data_disk.id
      mode            = "read-only"
      attachment_type = "data"
    }
  ]

  ssh_key        = local.my_ssh_key
  startup_script = <<-EOF
    #!/bin/bash
    git clone https://github.com/cedana/cedana-samples.git
    cd cedana-samples/scripts
    ./bootstrap-instance
  EOF
}

resource "crusoe_compute_instance" "cedana-demo-b" {
  name     = "cedana-demo-b"
  type     = "a10.1x"
  location = "us-northcentral1-a"

  # specify the base image
  image = "ubuntu22.04:latest"

  disks = [
    // disk attached at startup
    {
      id              = crusoe_storage_disk.data_disk.id
      mode            = "read-only"
      attachment_type = "data"
    }
  ]

  ssh_key        = local.my_ssh_key
  startup_script = <<-EOF
    #!/bin/bash
    git clone https://github.com/cedana/cedana-samples.git
    cd cedana-samples/scripts
    ./bootstrap-instance
  EOF
}

resource "crusoe_storage_disk" "data_disk" {
  name     = "data-disk"
  size     = "200GiB"
  location = "us-northcentral1-a"
}


output "instance_a_ip" {
  value = crusoe_compute_instance.instance_a.network_interface[0].access_config[0].nat_ip
}

output "instance_b_ip" {
  value = crusoe_compute_instance.instance_b.network_interface[0].access_config[0].nat_ip
}
