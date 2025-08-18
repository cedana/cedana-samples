import logging
import shutil
import sys
import signal
import os
from pathlib import Path
import numpy as np

from chai_lab.chai1 import run_inference

# Globals for cache + CR handshake
cache_dir = None
PERSIST_CACHE = False
wait_for_cr = False
cr_log_file = "/tmp/cr.log"

def inference(model=None):
    global cache_dir
    try:
        logging.basicConfig(level=logging.INFO)

        example_fasta = """>protein|name=example-of-long-protein
AGSHSMRYFSTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRGEPRAPWVEQEGPEYWDRETQKYKRQAQTDRVSLRNLRGYYNQSEAGSHTLQWMFGCDLGPDGRLLRGYDQSAYDGKDYIALNEDLRSWTAADTAAQITQRKWEAAREAEQRRAYLEGTCVEWLRRYLENGKETLQRAEHPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQWDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPEPLTLRWEP
>protein|name=example-of-short-protein
AIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM
>protein|name=example-peptide
GAAL
>ligand|name=example-ligand-as-smiles
CCCCCCCCCCCCCC(=O)O
""".strip()

        fasta_path = Path("/tmp/example.fasta")
        fasta_path.write_text(example_fasta)

        output_dir = Path("/tmp/outputs")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(exist_ok=True)

        # --- First inference (pre-CR) ---
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device="cuda:0",
            use_esm_embeddings=True,
        )
        print("Pre-CR scores:", [rd.aggregate_score.item() for rd in candidates.ranking_data])

        # --- Wait for CR ---
        if wait_for_cr:
            print("", flush=True)
            with open(cr_log_file, "a") as cr_log:
                print("CHECKPOINT", file=cr_log, flush=True)
            with open(cr_log_file, "r") as cr_log:
                while True:
                    line = cr_log.readline()
                    if "RESTORE" in line:
                        break

        # --- Second inference (post-CR) ---
        output_dir2 = Path("/tmp/outputs2")
        if output_dir2.exists():
            shutil.rmtree(output_dir2)
        output_dir2.mkdir(exist_ok=True)

        candidates2 = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir2,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device="cuda:0",
            use_esm_embeddings=True,
        )
        print("Post-CR scores:", [rd.aggregate_score.item() for rd in candidates2.ranking_data])

    except Exception as e:
        raise e
    finally:
        cleanup()
        if wait_for_cr:
            with open(cr_log_file, "a") as cr_log:
                print("DONE", file=cr_log, flush=True)

def usage():
    print(f"Usage: {sys.argv[0]} <model>")
    sys.exit(1)

def cleanup():
    global cache_dir
    if not PERSIST_CACHE:
        for d in ["/tmp/outputs", "/tmp/outputs2"]:
            shutil.rmtree(d, ignore_errors=True)

def handle_exit(signum, frame):
    sys.exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    if len(sys.argv) < 2:
        usage()
    model = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            model = arg
        elif arg == "--wait-for-cr":
            wait_for_cr = True
        elif arg == "--persist-cache":
            PERSIST_CACHE = True
    if model is None:
        usage()
    inference(model)
