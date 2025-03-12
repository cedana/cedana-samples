# Terraform for setting up Demo nodes
Running terraform init and apply inside each of these folders should bootstrap two nodes for you. 

## Setup 
Install the terraform CLI (https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli).

You need to have the following environment variables set in `~/.crusoe/config` (you'll need something similar for lambda). 

``` yaml
[default]
access_key_id="MY_ACCESS_KEY"
secret_key="MY_SECRET_KEY"


In addition to this, ensure you have a valid `terraform.tfvars` file in the provider of choice. 

```


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


