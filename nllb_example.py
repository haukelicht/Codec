from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",  src_lang="eng_Latn")
    
print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained("ychenNLP/nllb-200-distilled-1.3b-easyproject")
model.cuda()

input_chunks = ["[ A translator ] always risks inadvertently introducing [ source-language words ] , grammar, or syntax into the target-language rendering."]
print("Start translation...")
output_result = []

batch_size = 1
for idx in tqdm(range(0, len(input_chunks), batch_size)):
    start_idx = idx
    end_idx = idx + batch_size
    inputs = tokenizer(input_chunks[start_idx: end_idx], padding=True, truncation=True, max_length=128, return_tensors="pt").to('cuda')

    with torch.no_grad():
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], 
                        max_length=128, num_beams=5, num_return_sequences=1, early_stopping=True)

    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    output_result.extend(output)
print(output_result)

