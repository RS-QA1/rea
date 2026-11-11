# Child-voice research inventory (2026-11-11)

A public GitHub file is not automatically licensed for voice cloning, redistribution or commercial use. Do not assume the same terms apply to code, model weights, and the recorded person's voice. For real children's recordings, use sources with clear authorization and respect no-identification / no-reidentification restrictions.

| Source | Actual child voice? | New dialogue | Language | Model/recording availability | Caveats |
| --- | --- | --- | --- | --- | --- |
| https://github.com/rishabhjain16/child_tts_fastpitch | Yes, trained on children's speech | Checkpoint-dependent | English | Fine-tuning / checkpoint link | MyST dataset licensing; unverified deploy |
| https://github.com/C3Imaging/ChildTTS | Yes, trained on children's speech | Potential with RTVC stack | English | Encoder/acoustic/vocoder checkpoints via Drive | MyST separately licensed; full pipeline needs validation |
| https://huggingface.co/Aniket2003333333/xtts-v2-child-voice-finetuned | Yes, fine-tuned on Samrómur child speech | Yes, validated English WAV | English | 2.3GB checkpoint + reference | Quality not acceptable to user; Coqui base license caveats |
| https://github.com/life2allsofts/kidify-ai | Childlike conversion **claim**, not a child-TTS checkpoint | Changes timbre of supplied speech | Cross-lingual capability not validated for Telugu | Seed-VC model checkpoints | Requires authorized target speech, GPU recommended |
| https://github.com/Plachtaa/seed-vc | Voice conversion, not inherently child-trained | New speech supplied as source | Multilingual conversion needs testing | Model checkpoints | Upstream archived, GPU recommended |
| https://github.com/nam1665/text_to_speech_kids_voice | Experimental kids dataset TTS | Checkpoint-dependent | English | Older Tacotron samples/checkpoints | Insufficient validated reliability |
| https://huggingface.co/datasets/language-and-voice-lab/samromur_children | Yes, recorded children ages 4–17 | Data only; not a TTS model | Icelandic | Open audio/transcripts CC BY 4.0 | Voice rights, attribution, no-identification requirements |
| https://catalog.ldc.upenn.edu/LDC2021S05 | Yes, recordings of children | Data only | English | Restricted/licensed corpus | Commercial license separately required |

## Why not bulk-copy random GitHub WAV files?
Copyright/license for repository code does **not** grant permission to clone the recorded speaker. Sources may contain real minors' voices, and dataset terms can restrict reproduction. This inventory is for model evaluation and licensed/authorized reference selection, not mass scraping or automatic cloning.

## Plan
1. Keep separate working Telugu voices; no changes to adult-production app.
2. Test new child-oriented **cartoon/helium** DSP on Telugu speech; clearly label as an approximation.
3. Evaluate child-trained or Seed-VC source only if model, checkpoint, speaker rights and compute are verified.
4. Never call the app a genuine child-speech model until audible side-by-side tests demonstrate that.
