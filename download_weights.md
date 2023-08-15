## Downloading pretrained weights

## Official LLAMA 2 weights

```bash
python download.py --repo_id meta-llama/Llama-2-7b-chat-hf --token your_hf_token
```

Get your token here https://huggingface.co/settings/tokens

Convert the weights to the Lit-LLaMA format:

```bash
python convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
```