#!/bin/bash
# source .venv/bin/activate
export HF_TOKEN=< your_token > # Log in to Hugging Face. Go to https://huggingface.co/settings/tokens. Copy the "read" token.
for i in {0..5}
do
  python3 klreg_train.py --id $i
  python3 klreg_continue.py --logfile ${i}_chat.txt --steps 3000000 --chat_len 256
  python3 klreg_continue.py --logfile cont_${i}_chat.txt --steps 3000000 --chat_len 512
  python3 gen_text.py --id $i --filestr chat_cont
  python3 gen_text.py --id $i --filestr chat_cont_cont
done
python3 gen_base_transcripts.py
export OPENAI_API_KEY=< your_token >
python3 gpt_compare.py