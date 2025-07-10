module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0" # Use a recent, stable version of the module

  cluster_name    = var.cluster_name
  cluster_version = var.k8s_version

  vpc_id     = var.vpc_id
  subnet_ids = var.subnet_ids

  cluster_endpoint_private_access = true

  eks_managed_node_groups = {
    # GPU Node Pool
    gpu_pool = {
      name           = var.gpu_nodepool_name
      instance_types = [var.gpu_instance_type]
      min_size       = 2
      max_size       = 3
      desired_size   = 2
      map_public_ip_on_launch = true
      ami_type = "CUSTOM"
      ami_id   = data.aws_ssm_parameter.eks_ubuntu_ami.value
      key_name = var.key_pair_name
      enable_dns_hostnames = true
    }

    cpu_pool = {
      name           = var.cpu_nodepool_name
      instance_types = [var.cpu_instance_type]
      min_size       = 2
      max_size       = 3
      desired_size   = 2
      map_public_ip_on_launch = true
      # Use the same custom Ubuntu AMI.
      ami_type = "CUSTOM"
      ami_id   = data.aws_ssm_parameter.eks_ubuntu_ami.value
      key_name = var.key_pair_name
      enable_dns_hostnames = true
    }
  }
}


data "aws_ssm_parameter" "eks_ubuntu_ami" {
  name = "/aws/service/canonical/ubuntu/eks/24.04/${var.k8s_version}/stable/current/amd64/hvm/ebs-gp3/ami-id"
}
