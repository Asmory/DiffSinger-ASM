#!/usr/bin/env python3
from pathlib import Path
import argparse,torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--vocab',type=int,default=80);ap.add_argument('--c',type=int,default=384);ap.add_argument('--layers',type=int,default=4);a=ap.parse_args();C=a.c
    torch.manual_seed(21);sd={}
    def r(shape,s=.03):return torch.randn(*shape)*s
    sd['model.fs2.txt_embed.weight']=r((a.vocab,C),C**-.5);sd['model.fs2.txt_embed.weight'][0].zero_();sd['model.fs2.dur_embed.weight']=r((C,1));sd['model.fs2.dur_embed.bias']=r((C,),.01)
    for i in range(a.layers):
        p=f'model.fs2.encoder.layers.{i}.op.'
        sd[p+'layer_norm1.weight']=torch.ones(C);sd[p+'layer_norm1.bias']=torch.zeros(C);sd[p+'self_attn.in_proj.weight']=r((3*C,C));sd[p+'self_attn.out_proj.weight']=r((C,C));sd[p+'layer_norm2.weight']=torch.ones(C);sd[p+'layer_norm2.bias']=torch.zeros(C);sd[p+'ffn.ffn_1.weight']=r((4*C,C,3),.02);sd[p+'ffn.ffn_1.bias']=r((4*C,),.01);sd[p+'ffn.ffn_2.weight']=r((C,4*C),.02);sd[p+'ffn.ffn_2.bias']=r((C,),.01)
    sd['model.fs2.encoder.layer_norm.weight']=torch.ones(C);sd['model.fs2.encoder.layer_norm.bias']=torch.zeros(C)
    sd['model.fs2.stretch_embed.1.weight']=r((4*C,C));sd['model.fs2.stretch_embed.1.bias']=r((4*C,),.01);sd['model.fs2.stretch_embed.3.weight']=r((C,4*C));sd['model.fs2.stretch_embed.3.bias']=r((C,),.01)
    sd['model.fs2.stretch_embed_rnn.weight_ih_l0']=r((3*C,C),.025);sd['model.fs2.stretch_embed_rnn.bias_ih_l0']=r((3*C,),.01);sd['model.fs2.stretch_embed_rnn.weight_hh_l0']=r((3*C,C),.025);sd['model.fs2.stretch_embed_rnn.bias_hh_l0']=r((3*C,),.01);sd['model.fs2.pitch_embed.weight']=r((C,1));sd['model.fs2.pitch_embed.bias']=r((C,),.01)
    a.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':sd},a.out);print(a.out)
if __name__=='__main__':main()
