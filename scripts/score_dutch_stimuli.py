import json
from pathlib import Path
import torch
import librosa
from torch.nn import functional as F
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM

)
from tqdm import tqdm

LANG = "dutch"
MODEL_ID = f"jonatasgrosman/wav2vec2-large-xlsr-53-{LANG}"

wav_files = list(Path("data/wav/nl_audios").iterdir())


# *** acoustic scores ***
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

with open(f"confidence_scores_{LANG}.tsv", "w") as f:
    for wav_file_path in tqdm(wav_files, desc="acoustic scors"):

        stimulus_id = wav_file_path.stem
        speech_array, sampling_rate = librosa.load(wav_file_path)
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        probs = F.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(logits, dim=-1)
        pred_scores = probs.gather(1, predicted_ids.unsqueeze(-1))[:, :, 0]

        def confidence_score(word_dict, index):            
            probs = pred_scores[index, word_dict["start_offset"]: word_dict["end_offset"]]                                     
            return round(torch.sum(probs).item() / (len(probs)), 4)

        output = processor.batch_decode(predicted_ids, output_word_offsets=True)

        confidence_scores = {d["word"]: confidence_score(d, 0) for d in output.word_offsets[0]}

        for word, score in confidence_scores.items():
            f.write(f"{stimulus_id}\t{word}\t{score}\n")

# *** lm scores ***
processor = Wav2Vec2ProcessorWithLM.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

with open(f"lm_scores_{LANG}.tsv", "w") as f:
    for wav_file_path in tqdm(wav_files, desc="lm scores"):
        
        stimulus_id = wav_file_path.stem
        speech_array, sampling_rate = librosa.load(wav_file_path)
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt")

        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        output = processor.batch_decode(logits.numpy(), output_word_offsets=True)
        
        confidence_scores = [score / len(t.split(" ")) for score, t in zip(output.lm_score, output.text)]
        
        f.write(f"{stimulus_id}\t{confidence_scores[0]}\n")
