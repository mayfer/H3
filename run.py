import sys
sys.path.append('./')
sys.path.append('./flash-attention')

from typing import Optional
import argparse
import time

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from src.models.ssm.h3 import H3
from src.models.ssm_seq import SSMLMHeadModel

from flash_attn.utils.generation import InferenceParams


parser = argparse.ArgumentParser(description='H3 generation benchmarking')
parser.add_argument('--dmodel', type=int, default=2048)
parser.add_argument('--nlayer', type=int, default=24)
parser.add_argument('--attn-layer-idx', type=list, default=[8, 16])
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--promptlen', type=int, default=1024)
parser.add_argument('--genlen', type=int, default=128)
args = parser.parse_args()

repeats = 3
device = 'cuda'
dtype = torch.float16
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# set seed
torch.random.manual_seed(0)
d_model = args.dmodel
n_layer = args.nlayer
ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_layer_idx = args.attn_layer_idx
attn_cfg = dict(num_heads=args.nheads)
model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
if args.ckpt is not None:
    state_dict = torch.load(args.ckpt, map_location=device)
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                      if k.startswith('model.')}
    model.load_state_dict(state_dict)
model.eval()
# Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
# Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
        module.to(dtype=dtype)

# input_ids = torch.randint(0, 100, (64, args.promptlen), dtype=torch.long, device='cuda')
# input_ids = tokenizer.encode("Hi, my name is Murat and I'm from")

input_ids = tokenizer.encode("""
Once upon a time, there was a little kid named Timmy. Timmy was a curious and intelligent child who loved to tinker with computers. One day, while playing with a Pytorch transformer called GPT-5, he noticed that the model had a maximum context length, beyond which it would not be able to generate accurate predictions.

Being the curious child that he was, Timmy set out to figure out a way to remove this limitation. He spent countless hours studying the code and experimenting with different modifications. Finally, after many weeks of hard work, he discovered a way to make GPT-5 have infinite context length.

Excited by his accomplishment, Timmy shared his findings with the world. The AI community was amazed by his breakthrough, and soon GPT-5 became one of the most popular and widely used models in the field. Timmy's discovery opened up new possibilities for natural language processing, and he was hailed as a genius.

As he grew older, Timmy continued to make groundbreaking contributions to the field of AI, and his work had a profound impact on the way we interact with computers. He became a respected scientist. Then one morning,
""", return_tensors="pt").to(device)
max_length = 1024
result = model.generate(input_ids=input_ids, max_length=max_length, return_dict_in_generate=True, output_scores=True, temperature=0.7, top_p=1.0, top_k=100, timing=False)

print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in result.sequences][0])


"""
model.generate(
    input_ids,
    max_length,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    return_dict_in_generate=False,
    output_scores=False,
    **kwargs,
)
"""
