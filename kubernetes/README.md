Samples for running Cedana in a Kubernetes cluster. 

For GPU training workloads, you can run: 
`NUM_EPOCHS=100 envsubst < cuda-pytorch-cifar100.yaml | kubectl create -f -` to directly change the num of epochs without touching the yaml. 
