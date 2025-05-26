# cedana-samples
Samples and example workloads for our documentation. See docs.cedana.ai for examples on how to use!

For Kubernetes examples, see examples in our helm-chart repo: https://github.com/cedana/cedana-helm-charts. 

You can also use the cedana/cedana-samples docker container to run the arbitrary samples in here (which is the recommended approach - this container has everything!). 

## Setup 
Highly recommend using the requirements.txt to set up a virtualenv. 

``` sh
python3 -m env venv 
```

``` sh
source env/bin/active && pip install -r requirements.txt 
```


## llama.cpp 
Llama.cpp is downloaded directly from its github repo and unzipped into /usr/local/bin. To run arbitrary models, run `llama-cli -hf HUGGINGFACEMODEL`. 
