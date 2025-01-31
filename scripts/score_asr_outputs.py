from pathlib import Path

from huggingsound import SpeechRecognitionModel
from minicons import scorer
import pandas as pd
import torch
from torch.nn.functional import softmax

configs = [
    # English
    ("english", "jonatasgrosman/wav2vec2-large-xlsr-53-english", "gpt2"),
    # German
    ("german", "jonatasgrosman/wav2vec2-large-xlsr-53-german", "benjamin/gerpt2")
]

wav_files = list(Path("data/wav/nl_audios").iterdir())

def run_asr_and_lm(lang, asr_model_name, lm_name, entropy=False):
    results = []
    asr_model = SpeechRecognitionModel(asr_model_name)
    transcriptions = asr_model.transcribe(wav_files)
    lm_scorer = scorer.IncrementalLMScorer(lm_name)

    for f, t in zip(wav_files, transcriptions):
        sequence_score = lm_scorer.sequence_score(t["transcription"], reduction = lambda x: -x.mean(0).item(), base_two=True)
        t_prefix, t_suffix = " ".join(t["transcription"].split()[:-1]), t["transcription"].split()[-1]
        conditional_score = lm_scorer.conditional_score(t_prefix, t_suffix, base_two=True)
        t_tri_prefix, t_tri_suffix = " ".join(t["transcription"].split()[:-3]), " ".join(t["transcription"].split()[-3:])
        conditional_score_trigram = lm_scorer.conditional_score(t_tri_prefix, t_tri_suffix)
        results.append([f.stem, t["transcription"], sequence_score[0], -1*conditional_score[0], -1*conditional_score_trigram[0]])
        pass

    # entropy
    if entropy:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        lm = AutoModelForCausalLM.from_pretrained(lm_name)
        for i, (f, t) in enumerate(zip(wav_files, transcriptions)):
            tokenized = tokenizer(t["transcription"], return_tensors="pt")
            # print(tokenized)
            with torch.no_grad():
                outputs = lm(**tokenized, labels=tokenized["input_ids"])
            probs = softmax(outputs.logits, dim=-1)
            entropies = torch.sum(probs * -1*torch.log2(probs), dim=-1)
            full_entropy = entropies.mean().item()
            trigram_entropy = entropies[:,-3:].mean().item()
            results[i].extend([full_entropy, trigram_entropy])


    out_df = pd.DataFrame(results, columns=["stimulus_name", "stimulus_text", "sentence_score", "conditional_score", "conditional_score_trigram", "entropy", "entropy_trigram"])

    out_df.to_csv(f"scored_asr_outputs_{lang}.tsv", sep="\t", index=None)

        

if __name__ == "__main__":
    for lang, asr_model_name, lm_name in configs:
        print(lang, asr_model_name, lm_name)
        run_asr_and_lm(lang, asr_model_name, lm_name, entropy=True)