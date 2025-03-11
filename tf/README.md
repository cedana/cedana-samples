# Terraform for setting up Demo nodes
Running terraform init and apply inside each of these folders should bootstrap two nodes for you. You need to have the following environment variables set in `~/.crusoe/config` 

``` yaml
[default]
access_key_id="MY_ACCESS_KEY"
secret_key="MY_SECRET_KEY"


In addition to this, ensure you have the following env vars set: 
