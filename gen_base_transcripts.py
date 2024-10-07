import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch
device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from klreg import init_prompt
from gen_text import save_or_append_string_list
import pickle

n_transcripts = 2048
transcript_length = 256

n_per_batch = 64
n_batches = n_transcripts // n_per_batch

mixtral_cache_dir = '/nas/ucb/mkcohen/kl_reg/mixtral_cache_dir/'
# mixtral_cache_dir = '/nas/ucb/mkcohen/kl_reg/mixtral_cache_dir_2/'
def get_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=mixtral_cache_dir)
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map=device, quantization_config=bnb_config, cache_dir=mixtral_cache_dir)
    return model, tokenizer

model, tokenizer = get_model_and_tokenizer()

model_inputs = tokenizer([init_prompt for _ in range(n_per_batch)], return_tensors="pt").to(device)

filename = "transcripts/mixtral_base_transcripts.pkl"
for _ in range(n_batches):
    generated_ids = model.generate(**model_inputs, max_new_tokens=transcript_length-model_inputs['input_ids'].shape[1], do_sample=True)
    transcript_batch = tokenizer.batch_decode(generated_ids)
    save_or_append_string_list(filename, transcript_batch)