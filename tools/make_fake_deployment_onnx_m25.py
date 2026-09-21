#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch, onnx
from onnx import helper,TensorProto,numpy_helper

def npf(t):return np.ascontiguousarray(t.detach().cpu().float().numpy(),np.float32)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    x=torch.load(a.checkpoint,map_location='cpu',weights_only=True);sd=x.get('state_dict',x)
    C=int(sd['model.fs2.dur_embed.bias'].numel());V=int(sd['model.fs2.txt_embed.weight'].shape[0])
    init=[];nodes=[];counter=[0]
    def ini(name,x):init.append(numpy_helper.from_array(np.asarray(x),name=name));return name
    def const(name,val,dtype=np.float32):
        arr=np.asarray(val,dtype=dtype);nodes.append(helper.make_node('Constant',[],[name],name=name.replace('_out',''),value=numpy_helper.from_array(arr)));return name
    def mm(path,w,b=None):
        wn=ini(f'anon_{counter[0]}',np.asarray(w,dtype=np.float32).T);counter[0]+=1;mo=path+'_mm';nodes.append(helper.make_node('MatMul',['dummy',wn],[mo],name=path+'/MatMul'))
        if b is not None:
            bn=ini(b[0],b[1]);ao=path+'_add';nodes.append(helper.make_node('Add',[bn,mo],[ao],name=path+'/Add'));return ao
        return mo
    # common dummy input merely makes structural graph parseable
    inputs=[helper.make_tensor_value_info(n,TensorProto.FLOAT if n not in ('tokens','languages','durations','steps') else TensorProto.INT64,None) for n in ['tokens','languages','durations','f0','breathiness','voicing','tension','gender','velocity','spk_embed','depth','steps']]
    inputs.append(helper.make_tensor_value_info('dummy',TensorProto.FLOAT,None))
    # FS2 embeddings/linear
    ini('fs2.txt_embed.weight',npf(sd['model.fs2.txt_embed.weight']));nodes.append(helper.make_node('Gather',['fs2.txt_embed.weight','tokens'],['txt'],name='/fs2/txt_embed/Gather'))
    mm('/fs2/dur_embed',npf(sd['model.fs2.dur_embed.weight']),('fs2.dur_embed.bias',npf(sd['model.fs2.dur_embed.bias'])))
    lang=np.zeros((5,C),np.float32);lang[1:]=np.arange(4*C,dtype=np.float32).reshape(4,C)*1e-5;ini('fs2.lang_embed.weight',lang);nodes.append(helper.make_node('Gather',['fs2.lang_embed.weight','languages'],['lang'],name='/fs2/lang_embed/Gather'))
    L=len({int(k.split('layers.')[1].split('.')[0]) for k in sd if 'model.fs2.encoder.layers.' in k and k.endswith('layer_norm1.weight')})
    for i in range(L):
        p=f'model.fs2.encoder.layers.{i}.op.';b=f'/fs2/encoder/layers.{i}/op/'
        ini(f'fs2.encoder.layers.{i}.op.layer_norm1.weight',npf(sd[p+'layer_norm1.weight']));ini(f'fs2.encoder.layers.{i}.op.layer_norm1.bias',npf(sd[p+'layer_norm1.bias']));nodes.append(helper.make_node('LayerNormalization',['dummy',f'fs2.encoder.layers.{i}.op.layer_norm1.weight',f'fs2.encoder.layers.{i}.op.layer_norm1.bias'],[f'ln1{i}'],name=b+'layer_norm1/LayerNormalization'))
        mm(b+'self_attn/in_proj',npf(sd[p+'self_attn.in_proj.weight']))
        mm(b+'self_attn/out_proj',npf(sd[p+'self_attn.out_proj.weight']))
        ini(f'fs2.encoder.layers.{i}.op.layer_norm2.weight',npf(sd[p+'layer_norm2.weight']));ini(f'fs2.encoder.layers.{i}.op.layer_norm2.bias',npf(sd[p+'layer_norm2.bias']));nodes.append(helper.make_node('LayerNormalization',['dummy',f'fs2.encoder.layers.{i}.op.layer_norm2.weight',f'fs2.encoder.layers.{i}.op.layer_norm2.bias'],[f'ln2{i}'],name=b+'layer_norm2/LayerNormalization'))
        fw=npf(sd[p+'ffn.ffn_1.weight']);fb=npf(sd[p+'ffn.ffn_1.bias']);ini(f'fs2.encoder.layers.{i}.op.ffn.ffn_1.weight',fw);ini(f'fs2.encoder.layers.{i}.op.ffn.ffn_1.bias',fb);nodes.append(helper.make_node('Conv',['dummy',f'fs2.encoder.layers.{i}.op.ffn.ffn_1.weight',f'fs2.encoder.layers.{i}.op.ffn.ffn_1.bias'],[f'conv{i}'],name=b+'ffn/ffn_1/Conv'))
        mm(b+'ffn/ffn_2',npf(sd[p+'ffn.ffn_2.weight']),(f'fs2.encoder.layers.{i}.op.ffn.ffn_2.bias',npf(sd[p+'ffn.ffn_2.bias'])))
    ini('fs2.encoder.layer_norm.weight',npf(sd['model.fs2.encoder.layer_norm.weight']));ini('fs2.encoder.layer_norm.bias',npf(sd['model.fs2.encoder.layer_norm.bias']));nodes.append(helper.make_node('LayerNormalization',['dummy','fs2.encoder.layer_norm.weight','fs2.encoder.layer_norm.bias'],['fln'],name='/fs2/encoder/layer_norm/LayerNormalization'))
    for j in (1,3):
        w=npf(sd[f'model.fs2.stretch_embed.{j}.weight']);b=npf(sd[f'model.fs2.stretch_embed.{j}.bias']);wn=ini(f'stretch{j}.w',w);bn=ini(f'stretch{j}.b',b);nodes.append(helper.make_node('Gemm',['dummy',wn,bn],[f'stretch{j}'],name=f'/fs2/stretch_embed/stretch_embed.{j}/Gemm',transB=1))
    # PyTorch r,z,n -> ONNX z,r,h
    def rzn2zrh(q):r,z,n=np.split(q,3,axis=0);return np.concatenate([z,r,n],0)
    W=rzn2zrh(npf(sd['model.fs2.stretch_embed_rnn.weight_ih_l0']))[None];R=rzn2zrh(npf(sd['model.fs2.stretch_embed_rnn.weight_hh_l0']))[None];bi=rzn2zrh(npf(sd['model.fs2.stretch_embed_rnn.bias_ih_l0']));bh=rzn2zrh(npf(sd['model.fs2.stretch_embed_rnn.bias_hh_l0']));B=np.concatenate([bi,bh])[None]
    ini('onnx_gru_w',W);ini('onnx_gru_r',R);ini('onnx_gru_b',B);nodes.append(helper.make_node('GRU',['dummy','onnx_gru_w','onnx_gru_r','onnx_gru_b'],['gru_o','gru_h'],name='/fs2/stretch_embed_rnn/GRU'))
    mm('/fs2/pitch_embed',npf(sd['model.fs2.pitch_embed.weight']),('fs2.pitch_embed.bias',npf(sd['model.fs2.pitch_embed.bias'])))
    # deployment features
    rng=np.random.default_rng(25)
    for idx,(nm,mul,scale) in enumerate([('breathiness',4,1/96),('voicing',5,1/96),('tension',6,.1)]):
        co=const(f'/fs2/Constant_feat_{nm}',np.array(scale,np.float32));nodes.append(helper.make_node('Mul',[nm,co],[f'{nm}_scaled'],name=f'/fs2/Mul_{mul}'));w=(rng.standard_normal((C,1))*0.02).astype(np.float32);bn=f'fs2.{nm}.bias';mm(f'/fs2/{nm}',w,(bn,np.zeros(C,np.float32)))
    cmin=const('/fs2/Constant_33_output_0',np.array(-1,np.float32));cmax=const('/fs2/Constant_4_output_0',np.array(1,np.float32));nodes.append(helper.make_node('Clip',['gender',cmin,cmax],['gclip'],name='/fs2/Clip'));c12=const('/fs2/Constant_37_output_0',np.array(12,np.float32));nodes.append(helper.make_node('Mul',['dummy',c12],['g7'],name='/fs2/Mul_7'));nodes.append(helper.make_node('Mul',['dummy',c12],['g8'],name='/fs2/Mul_8'));nodes.append(helper.make_node('Add',['g7','g8'],['gadd'],name='/fs2/Add_7'));nodes.append(helper.make_node('Mul',['gclip','gadd'],['g9'],name='/fs2/Mul_9'));c112=const('/fs2/Constant_40_output_0',np.array(1/12,np.float32));nodes.append(helper.make_node('Mul',['g9',c112],['g10'],name='/fs2/Mul_10'));mm('/fs2/key_shift_embed',(rng.standard_normal((C,1))*0.02).astype(np.float32),('fs2.key_shift_embed.bias',np.zeros(C,np.float32)))
    smin=const('/fs2/Constant_41_output_0',np.array(.1,np.float32));smax=const('/fs2/Constant_42_output_0',np.array(5,np.float32));nodes.append(helper.make_node('Clip',['velocity',smin,smax],['vclip'],name='/fs2/Clip_1'));mm('/fs2/speed_embed',(rng.standard_normal((C,1))*0.02).astype(np.float32),('fs2.speed_embed.bias',np.zeros(C,np.float32)))
    # aux
    pref='model.aux_decoder.decoder.'
    for stem in ('inconv','outconv'):
        w=npf(sd[pref+stem+'.weight']);b=npf(sd[pref+stem+'.bias']);ini(f'aux_decoder.decoder.{stem}.weight',w);ini(f'aux_decoder.decoder.{stem}.bias',b);nodes.append(helper.make_node('Conv',['dummy',f'aux_decoder.decoder.{stem}.weight',f'aux_decoder.decoder.{stem}.bias'],[stem],name=f'/aux_decoder/decoder/{stem}/Conv'))
    AL=len({int(k.split('conv.')[1].split('.')[0]) for k in sd if 'model.aux_decoder.decoder.conv.' in k and k.endswith('dwconv.weight')})
    for i in range(AL):
        p=f'model.aux_decoder.decoder.conv.{i}.';b=f'/aux_decoder/decoder/conv.{i}/'
        for n in ('dwconv.weight','dwconv.bias','norm.weight','norm.bias','gamma'):ini('aux_decoder.decoder.conv.%d.%s'%(i,n),npf(sd[p+n]))
        nodes.append(helper.make_node('Conv',['dummy',f'aux_decoder.decoder.conv.{i}.dwconv.weight',f'aux_decoder.decoder.conv.{i}.dwconv.bias'],[f'adw{i}'],name=b+'dwconv/Conv'));nodes.append(helper.make_node('LayerNormalization',['dummy',f'aux_decoder.decoder.conv.{i}.norm.weight',f'aux_decoder.decoder.conv.{i}.norm.bias'],[f'aln{i}'],name=b+'norm/LayerNormalization'))
        mm(b+'pwconv1',npf(sd[p+'pwconv1.weight']),(f'aux_decoder.decoder.conv.{i}.pwconv1.bias',npf(sd[p+'pwconv1.bias'])));mm(b+'pwconv2',npf(sd[p+'pwconv2.weight']),(f'aux_decoder.decoder.conv.{i}.pwconv2.bias',npf(sd[p+'pwconv2.bias'])));nodes.append(helper.make_node('Mul',[f'aux_decoder.decoder.conv.{i}.gamma','dummy'],[f'ag{i}'],name=b+'Mul'))
    k=const('/aux_decoder/Constant_output_0',np.array([6.],np.float32));bb=const('/aux_decoder/Constant_1_output_0',np.array([-6.],np.float32));nodes.append(helper.make_node('Mul',['dummy',k],['auxmul'],name='/aux_decoder/Mul'));nodes.append(helper.make_node('Add',['auxmul',bb],['aux_mel'],name='/aux_decoder/Add'))
    # rf under velocity_fn names
    rp='model.diffusion.denoise_fn.'
    def rname(s):return 'diffusion.velocity_fn.'+s
    for lin in ('input_projection','diffusion_embedding.1','diffusion_embedding.3','output_projection'):
        mm('/diffusion/velocity_fn/'+lin.replace('.','/'),npf(sd[rp+lin+'.weight']),(rname(lin+'.bias'),npf(sd[rp+lin+'.bias'])))
    w=npf(sd[rp+'conditioner_projection.weight']);b=npf(sd[rp+'conditioner_projection.bias']);ini(rname('conditioner_projection.weight'),w);ini(rname('conditioner_projection.bias'),b);nodes.append(helper.make_node('Conv',['dummy',rname('conditioner_projection.weight'),rname('conditioner_projection.bias')],['rc'],name='/diffusion/velocity_fn/conditioner_projection/Conv'))
    ini(rname('norm.weight'),npf(sd[rp+'norm.weight']));ini(rname('norm.bias'),npf(sd[rp+'norm.bias']))
    RL=len({int(k.split('residual_layers.')[1].split('.')[0]) for k in sd if rp+'residual_layers.' in k})
    for i in range(RL):
        for j in (0,2):
            for z in ('weight','bias'):ini(rname(f'residual_layers.{i}.net.{j}.{z}'),npf(sd[rp+f'residual_layers.{i}.net.{j}.{z}']))
        nodes.append(helper.make_node('LayerNormalization',['dummy',rname(f'residual_layers.{i}.net.0.weight'),rname(f'residual_layers.{i}.net.0.bias')],[f'rln{i}'],name=f'/diffusion/velocity_fn/residual_layers.{i}/net.0/LayerNormalization'))
        nodes.append(helper.make_node('Conv',['dummy',rname(f'residual_layers.{i}.net.2.weight'),rname(f'residual_layers.{i}.net.2.bias')],[f'rdw{i}'],name=f'/diffusion/velocity_fn/residual_layers.{i}/net.2/Conv'))
        for j in (4,6,8):mm(f'/diffusion/velocity_fn/residual_layers.{i}/net.{j}',npf(sd[rp+f'residual_layers.{i}.net.{j}.weight']),(rname(f'residual_layers.{i}.net.{j}.bias'),npf(sd[rp+f'residual_layers.{i}.net.{j}.bias'])))
    nodes.append(helper.make_node('Atan',['dummy'],['atan'],name='/diffusion/velocity_fn/residual_layers.0/atan'))
    out=helper.make_tensor_value_info('aux_mel',TensorProto.FLOAT,None);g=helper.make_graph(nodes,'m25fake',inputs,[out],initializer=init);m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17)]);a.out.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,a.out);print(a.out)
if __name__=='__main__':main()
