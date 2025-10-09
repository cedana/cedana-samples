#!/usr/bin/env python3
import signal
import sys

import nemo.collections.asr as nemo_asr


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name='nvidia/parakeet-ctc-1.1b'
)
asr_model.transcribe(['sample.wav'])
