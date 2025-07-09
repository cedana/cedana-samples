# --------------------------------------------------------------------------------------------------
# main.tf - Main configuration for the EKS Cluster and Node Groups
# --------------------------------------------------------------------------------------------------

provider "aws" {
  region = var.aws_region
}

# --- Data Source for Canonical Ubuntu 22.04 AMI ---
# Uses the SSM Parameter path for official Canonical Ubuntu AMIs for EKS.
data "aws_ssm_parameter" "eks_ubuntu_ami_id" {
  name = "/aws/service/canonical/ubuntu/eks/24.04/${var.k8s_version}/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

# --- IAM Role for EKS Cluster ---
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

# --- EKS Cluster ---
resource "aws_eks_cluster" "cedana_ci_cluster" {
  name     = var.cluster_name
  version  = var.k8s_version
  role_arn = aws_iam_role.eks_cluster_role.arn

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = {
    Name = var.cluster_name
  }
}

# --- IAM Role for EKS Node Groups ---
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

# --- Launch Templates for Node Groups ---
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

# --- GPU Node Group ---
resource "aws_eks_node_group" "gpu_node_group" {
  cluster_name    = aws_eks_cluster.cedana_ci_cluster.name
  node_group_name = var.gpu_nodepool_name
  node_role_arn   = aws_iam_role.eks_node_group_role.arn
  subnet_ids      = var.subnet_ids

  launch_template {
    id      = aws_launch_template.gpu_nodes_lt.id
    version = aws_launch_template.gpu_nodes_lt.latest_version
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

# --- CPU Node Group ---
resource "aws_eks_node_group" "cpu_node_group" {
  cluster_name    = aws_eks_cluster.cedana_ci_cluster.name
  node_group_name = var.cpu_nodepool_name
  node_role_arn   = aws_iam_role.eks_node_group_role.arn
  subnet_ids      = var.subnet_ids

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

# --------------------------------------------------------------------------------------------------
# Kubernetes Provider and Auth Configuration
# --------------------------------------------------------------------------------------------------

provider "kubernetes" {
  host                   = aws_eks_cluster.cedana_ci_cluster.endpoint
  cluster_ca_certificate = base64decode(aws_eks_cluster.cedana_ci_cluster.certificate_authority[0].data)

  # This exec block configures the provider to use 'aws eks get-token' to authenticate.
  # This is the standard way to connect Terraform to an EKS cluster.
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", aws_eks_cluster.cedana_ci_cluster.name]
  }
}

# This resource creates the aws-auth ConfigMap in the kube-system namespace.
# This ConfigMap is critical for allowing IAM roles (like our node group role)
# to authenticate with the Kubernetes API server.
resource "kubernetes_config_map" "aws_auth" {
  # This depends_on is crucial. It ensures the cluster and node groups are created
  # before Terraform tries to apply this Kubernetes configuration.
  depends_on = [aws_eks_node_group.cpu_node_group, aws_eks_node_group.gpu_node_group]

  metadata {
    name      = "aws-auth"
    namespace = "kube-system"
  }

  data = {
    # The mapRoles key maps IAM roles to Kubernetes user and group permissions.
    mapRoles = yamlencode([
      {
        # This maps the IAM role used by our EC2 nodes.
        rolearn = aws_iam_role.eks_node_group_role.arn
        # This username template is required by EKS.
        username = "system:node:{{EC2PrivateDNSName}}"
        # These groups grant the necessary permissions for a node to join the cluster.
        groups = [
          "system:bootstrappers",
          "system:nodes",
        ]
      }
    ])
    # The mapUsers key can be used to map IAM users, but we don't need it for nodes.
    mapUsers = yamlencode([])
  }
}

