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

## Host memory samples (`cpu_smr/hostmem/`)
Checkpoint/restore regression workloads for shared memory, ghost files (files unlinked while still mapped) and IPC. Python 3 stdlib only, no root needed. Each logs `READY` once set up, progress lines while running, and `MISMATCH ...` (then exits non-zero) on any error. With `--gate PATH` each runs until that file exists, checks once more, logs `PASS` and exits 0, so a test can checkpoint and restore it mid-run and then create the gate.

| Sample | What it exercises |
|---|---|
| `hostmem.py` | Memory regions of every kind: private anonymous, `MAP_SHARED` anonymous, a named `/dev/shm` file (kept, or unlinked as a ghost), a memfd, a `MAP_SHARED` regular file (kept or unlinked), and a sparse file over 4 GiB. Each has data pages, pages written as zeros, holes, zero pages with a few non-zero bytes at either end, and a partly written last page (zero-page elision, short reads at EOF, ghost files over 4 GiB, parallel shmem dumps). `--procs` children keep verifying every region and counting in each shared region's comm page. The parent logs `OK <count per child>` and checks the counts keep advancing (the mappings are still shared after a restore) and that writes to private memory stay private. |
| `shm-messages.py` | A producer and a consumer passing messages through POSIX shared memory slots and a `multiprocessing.Queue` (a pipe and POSIX semaphores in `/dev/shm`), as an inference server's processes do. Every message must arrive, in order, with its payload. Logs `SEQ <n>`. |
| `ghost-busy-name.py` | Processes mapping a file that is then replaced under the same name (`os.replace`), as tensor-parallel ranks do with a JIT-compiled library: a ghost file whose name is busy with another file. |

``` sh
# Every kind of region, small; or --preset large|sparse|hicache, or your own:
cpu_smr/hostmem/hostmem.py --preset small
cpu_smr/hostmem/hostmem.py --regions anon:64M,shm:256M,ghost-shm:1G,memfd:128M,ghost-sparse:5G \
    --procs 4 --dir /tmp --gate /tmp/gate --log /tmp/hostmem.log
# KIND:SIZE:sparse or :dense picks the layout (sparse kinds are sparse by default)
cpu_smr/hostmem/hostmem.py --regions ghost-shm:32M:sparse

cpu_smr/hostmem/shm-messages.py /tmp/msgs.log [--gate /tmp/gate]
cpu_smr/hostmem/ghost-busy-name.py /tmp/dir /tmp/ghost.log [--gate /tmp/gate]
```

`/dev/shm` regions that don't fit in `/dev/shm` (64 MiB by default in Docker) are shrunk to fit, or skipped, with a `NOTE` line.
