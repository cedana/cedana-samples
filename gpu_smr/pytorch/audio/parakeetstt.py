#!/usr/bin/env python3
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")
asr_model.transcribe(['sample.wav'])
