provider "aws" {
  region = var.aws_region
}

data "aws_ssm_parameter" "eks_ubuntu_ami_id" {
  name = "/aws/service/eks/optimized-ami/${var.k8s_version}/ubuntu-22.04/amd64/recommended/image_id"
}

data "aws_ssm_parameter" "eks_ubuntu_ami_release_version" {
  name = "/aws/service/eks/optimized-ami/${var.k8s_version}/ubuntu-22.04/amd64/recommended/release_version"
}

resource "aws_vpc" "eks_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

resource "aws_subnet" "eks_subnet" {
  count                   = 2
  vpc_id                  = aws_vpc.eks_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true # For simplicity, we'll use public subnets. For production, private subnets are recommended.

  tags = {
    Name                                        = "${var.cluster_name}-subnet-${count.index + 1}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_iam_role" "eks_cluster_role" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "eks.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${var.cluster_name}-cluster-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

resource "aws_eks_cluster" "cedana_ci_cluster" {
  name     = var.cluster_name
  version  = var.k8s_version
  role_arn = aws_iam_role.eks_cluster_role.arn

  vpc_config {
    subnet_ids = [for subnet in aws_subnet.eks_subnet : subnet.id]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = {
    Name = var.cluster_name
  }
}

resource "aws_iam_role" "eks_node_group_role" {
  name = "${var.cluster_name}-node-group-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Name = "${var.cluster_name}-node-group-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_group_role.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_group_role.name
}

resource "aws_iam_role_policy_attachment" "ec2_container_registry_read_only" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_group_role.name
}

resource "aws_launch_template" "gpu_nodes_lt" {
  name_prefix   = "${var.gpu_nodepool_name}-"
  image_id      = data.aws_ssm_parameter.eks_ubuntu_ami_id.value
  instance_type = var.gpu_instance_type

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = var.gpu_nodepool_name
    }
  }
}

resource "aws_launch_template" "cpu_nodes_lt" {
  name_prefix   = "${var.cpu_nodepool_name}-"
  image_id      = data.aws_ssm_parameter.eks_ubuntu_ami_id.value
  instance_type = var.cpu_instance_type

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = var.cpu_nodepool_name
    }
  }
}

resource "aws_eks_node_group" "gpu_node_group" {
  cluster_name    = aws_eks_cluster.cedana_ci_cluster.name
  node_group_name = var.gpu_nodepool_name
  node_role_arn   = aws_iam_role.eks_node_group_role.arn
  subnet_ids      = [for subnet in aws_subnet.eks_subnet : subnet.id]
  release_version = data.aws_ssm_parameter.eks_ubuntu_ami_release_version.value

  launch_template {
    id      = aws_launch_template.gpu_nodes_lt.id
    version = aws_launch_template.gpu_nodes_lt.latest_version
  }

  scaling_config {
    desired_size = 2
    max_size     = 3 # Allows for rolling updates
    min_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    "nodepool-name" = var.gpu_nodepool_name,
    "instance-type" = "gpu"
  }

  tags = {
    Name = var.gpu_nodepool_name
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read_only,
  ]
}

resource "aws_eks_node_group" "cpu_node_group" {
  cluster_name    = aws_eks_cluster.cedana_ci_cluster.name
  node_group_name = var.cpu_nodepool_name
  node_role_arn   = aws_iam_role.eks_node_group_role.arn
  subnet_ids      = [for subnet in aws_subnet.eks_subnet : subnet.id]
  release_version = data.aws_ssm_parameter.eks_ubuntu_ami_release_version.value

  launch_template {
    id      = aws_launch_template.cpu_nodes_lt.id
    version = aws_launch_template.cpu_nodes_lt.latest_version
  }

  scaling_config {
    desired_size = 2
    max_size     = 3
    min_size     = 2
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    "nodepool-name" = var.cpu_nodepool_name,
    "instance-type" = "cpu"
  }

  tags = {
    Name = var.cpu_nodepool_name
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read_only,
  ]
}

