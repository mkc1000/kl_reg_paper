# KL Regularization

## Setup

1. Install Packages:

    a. Run `pip3 install -r requirements.txt`.

2. Get a Hugging Face Token:

    a. Log in to Hugging Face.

    b. Go to https://huggingface.co/mistralai/Mixtral-8x7B-v0.1, and request permission to use the model.

    c. Go to https://huggingface.co/settings/tokens.

    d. Copy the "read" token.

3. Add your Hugging Face Token to the script:

    a. Go to run.sh, and paste the token just after `export HF_TOKEN=`.

## Running

1. Acquire two 80GB GPUs, ideally A100-SXM4-80GB. PyTorch will be excpeting two devices called "cuda:0" and "cuda:1".

2. Run run.sh.

3. Output can be monitored in the stdout folder, and in the file stderr.txt.

