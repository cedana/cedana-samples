# Terraform for setting up Demo nodes
Running terraform init and apply inside each of these folders should bootstrap two nodes for you. 

## Setup 
Install the terraform CLI (https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).

You need to have the following environment variables set in `~/.crusoe/config` (you'll need something similar for lambda). 

``` yaml
[default]
access_key_id="MY_ACCESS_KEY"
secret_key="MY_SECRET_KEY"

```


In addition to this, ensure you have a valid `terraform.tfvars` file in the provider of choice. Take a look at variables.tf to see what you need to add. An example of a terraform.tfvars file is shown below: 

``` sh
cedana_url            = "https://something.run/v1"
cedana_auth_token     = "MY_AUTH_TOKEN"
aws_access_key_id     = "MY_AWS_KEY"
aws_secret_access_key = "MY_AWS_SECRET_ACCESS_KEY"
ssh_key               = "~/.ssh/my-public-key.pub"
instance_names        = ["instance-a", "instance-b"]
```

To create an SSH key, follow the "generating a new SSH key" section on [Github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) . 

If using Crusoe, I recommend installing the Crusoe [CLI](https://docs.crusoecloud.com/reference/cli/).

### Creating nodes 

``` sh
terraform init
```

``` sh
terraform apply -var-file=terraform.tfvars
```

### Destroying nodes & cleanup 

``` sh
terraform destroy -var-file=terraform.tfvars
```

## Node setup 
SSH into the node and run `./cedana-samples/scripts/bootstrap-instance`. After it completes, you should have a `start.sh` file in your root folder, which you can run to mount an s3 bucket and run cedana. 

