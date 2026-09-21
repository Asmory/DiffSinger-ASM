#!/usr/bin/env python3
"""Pack an exported DiffSinger acoustic.onnx into DiffSinger-ASM M25 bundles.

Targeted at current OpenVPI deployment graphs.  Linear weights are recovered
from graph topology, so anonymous exporter names such as onnx::MatMul_1651 are
not part of the format contract.
"""
from __future__ import annotations
import argparse,json,math,re,shutil,struct
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper,AttributeProto

ALIGN=64
LANG=1<<0;BREATH=1<<1;VOICE=1<<2;TENSION=1<<3;KEY=1<<4;SPEED=1<<5;SPK=1<<6;STRETCH=1<<7;LANGMASK=1<<8

def al(x):return (x+63)//64*64
def pack16(w):
    w=np.ascontiguousarray(w,np.float32);n,k=w.shape
    if n%16:raise ValueError(f'packed16 N must be multiple of 16, got {w.shape}')
    return np.ascontiguousarray(w.reshape(n//16,16,k).transpose(0,2,1)).ravel()
def conv_flat(w):return np.ascontiguousarray(w.transpose(0,2,1).reshape(w.shape[0],-1),np.float32)
def f32(x):return np.ascontiguousarray(np.asarray(x,dtype=np.float32))

class Graph:
    def __init__(self,m):
        self.m=m;self.nodes=[];self.init={};self.prod={};self.const={};self.inputs={x.name for x in m.graph.input}
        self._walk(m.graph)
    def _walk(self,g):
        for t in g.initializer:self.init[t.name]=numpy_helper.to_array(t)
        for n in g.node:
            self.nodes.append(n)
            for o in n.output:self.prod[o]=n
            if n.op_type=='Constant' and n.output:
                for a in n.attribute:
                    if a.name=='value' and a.type==AttributeProto.TENSOR:self.const[n.output[0]]=numpy_helper.to_array(a.t)
                    elif a.name=='value_float':self.const[n.output[0]]=np.array(a.f,np.float32)
                    elif a.name=='value_int':self.const[n.output[0]]=np.array(a.i,np.int64)
            for a in n.attribute:
                if a.type==AttributeProto.GRAPH:self._walk(a.g)
                elif a.type==AttributeProto.GRAPHS:
                    for q in a.graphs:self._walk(q)
    def value(self,name):
        if name in self.init:return np.asarray(self.init[name])
        if name in self.const:return np.asarray(self.const[name])
        raise KeyError(f'constant/initializer not found: {name}')
    def scalar(self,name):
        x=self.value(name).reshape(-1)
        if x.size!=1:raise ValueError(f'expected scalar {name}, got {x.shape}')
        return float(x[0])
    def find(self,frag,op=None):
        # Prefer an exact exported node name.  ONNX exporters commonly emit
        # siblings such as /fs2/Clip and /fs2/Clip_1; substring-only lookup
        # makes the shorter name spuriously ambiguous.
        exact=[n for n in self.nodes if n.name==frag and (op is None or n.op_type==op)]
        if len(exact)==1:
            return exact[0]
        if len(exact)>1:
            raise KeyError(f'exact node {frag!r} op={op}: {len(exact)} matches')
        q=[n for n in self.nodes if frag in n.name and (op is None or n.op_type==op)]
        if len(q)!=1:
            raise KeyError(f'node fragment {frag!r} op={op}: {len(q)} matches: {[n.name for n in q[:12]]}')
        return q[0]
    def find_re(self,pat,op=None):
        rx=re.compile(pat);q=[n for n in self.nodes if rx.search(n.name) and (op is None or n.op_type==op)]
        return q
    def matmul_weight(self,node):
        if node.op_type=='MatMul':b=self.value(node.input[1]);return f32(b.T)
        if node.op_type=='Gemm':
            b=self.value(node.input[1]); attrs={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute};tb=int(attrs.get('transB',0))
            return f32(b if tb else b.T)
        raise ValueError(node.op_type)
    def linear_from_frag(self,frag,bias_frag=None):
        # Exact-name first for the same reason as find(); fall back to a
        # fragment only when the exporter did not preserve the exact path.
        q=[n for n in self.nodes if n.name==frag and n.op_type in ('MatMul','Gemm')]
        if not q:
            q=[n for n in self.nodes if frag in n.name and n.op_type in ('MatMul','Gemm')]
        if len(q)!=1:raise KeyError(f'linear {frag}: {[n.name for n in q[:12]]}')
        n=q[0];w=self.matmul_weight(n);b=None
        if n.op_type=='Gemm' and len(n.input)>2 and n.input[2]:b=f32(self.value(n.input[2])).reshape(-1)
        if b is None and bias_frag:
            cand=[name for name in self.init if bias_frag in name and name.endswith('bias')]
            if len(cand)==1:b=f32(self.value(cand[0])).reshape(-1)
        if b is None:b=np.zeros(w.shape[0],np.float32)
        return w,b
    def linear_from_bias(self,bias_name):
        # Exact or unique suffix initializer name -> Add/Gemm consumer -> upstream MatMul.
        keys=[k for k in self.init if k==bias_name or k.endswith(bias_name)]
        if len(keys)!=1:raise KeyError(f'bias {bias_name}: {keys[:12]}')
        bn=keys[0];cons=[n for n in self.nodes if bn in n.input and n.op_type in ('Add','Gemm')]
        if len(cons)!=1:raise KeyError(f'consumer of {bn}: {[n.name for n in cons]}')
        c=cons[0];b=f32(self.value(bn)).reshape(-1)
        if c.op_type=='Gemm':return self.matmul_weight(c),b
        other=next(x for x in c.input if x!=bn);p=self.prod.get(other)
        if p is None or p.op_type!='MatMul':raise KeyError(f'{bn}: upstream MatMul not found, other={other}')
        return self.matmul_weight(p),b
    def linear_by_node(self,frag,bias_suffix=None):
        q=[n for n in self.nodes if frag in n.name and n.op_type in ('MatMul','Gemm')]
        if not q:raise KeyError(f'linear node fragment {frag!r}: no MatMul/Gemm')
        # If the same body was duplicated into ONNX If branches, exporter names and
        # parameters should still identify the same logical layer.
        w0=self.matmul_weight(q[0])
        for n in q[1:]:
            w=self.matmul_weight(n)
            if w.shape!=w0.shape or not np.array_equal(w,w0):
                raise KeyError(f'linear node fragment {frag!r}: ambiguous {[x.name for x in q[:8]]}')
        b=None
        if q[0].op_type=='Gemm' and len(q[0].input)>2 and q[0].input[2]:
            b=f32(self.value(q[0].input[2])).reshape(-1)
        if b is None and bias_suffix:
            ks=[k for k in self.init if k==bias_suffix or k.endswith(bias_suffix)]
            if len(ks)==1:b=f32(self.value(ks[0])).reshape(-1)
        if b is None:b=np.zeros(w0.shape[0],np.float32)
        return w0,b

def write_blob(path,header,arrays):
    secs={};path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:
        assert len(header)==64;f.write(header);off=64
        for name,x in arrays:
            x=f32(x);ao=al(off);f.write(b'\0'*(ao-off));off=ao;raw=x.tobytes();f.write(raw);secs[name]={'offset':off,'bytes':len(raw),'shape':list(x.shape)};off+=len(raw)
    return secs

def pack_fs2(G,out):
    emb=f32(G.value(G.find('/fs2/txt_embed/Gather','Gather').input[0]));V,C=emb.shape
    dw,db=G.linear_from_frag('/fs2/dur_embed/MatMul','fs2.dur_embed.bias');
    if dw.shape!=(C,1):raise ValueError(('dur',dw.shape))
    lang=None;nlang=0;flags=0
    lang_mask=None;cross_ids=[]
    if 'languages' in G.inputs:
        lang=f32(G.value(G.find('/fs2/lang_embed/Gather','Gather').input[0]));nlang=lang.shape[0];flags|=LANG
        if lang.shape[1]!=C:raise ValueError(('lang',lang.shape,C))
        # FastSpeech2AcousticONNX applies language IDs only to tokens listed in
        # cross_lingual_token_idx: lang_embed(languages * any(tokens[...,None] == ids)).
        # Recover that exported constant directly from /fs2/Equal.
        eq=G.find('/fs2/Equal','Equal')
        cands=[z for z in eq.input if z in G.init or z in G.const]
        if len(cands)!=1:
            raise KeyError(f'cross-lingual token constant for /fs2/Equal: inputs={list(eq.input)} constants={cands}')
        ids=np.asarray(G.value(cands[0])).astype(np.int64).reshape(-1)
        cross_ids=sorted({int(v) for v in ids if 0 <= int(v) < V})
        if not cross_ids:
            raise ValueError(f'languages input exists but exported cross_lingual_token_idx is empty: {ids.tolist()}')
        lang_mask=np.zeros(V,np.float32);lang_mask[cross_ids]=1.0;flags|=LANGMASK
    ids=sorted({int(m.group(1)) for n in G.nodes for m in [re.search(r'/fs2/encoder/layers\.(\d+)/op/layer_norm1/LayerNormalization$',n.name)] if m})
    if ids!=list(range(len(ids))) or not ids:raise ValueError(f'FS2 layers {ids}')
    arr=[];add=lambda n,x:arr.append((n,f32(x)))
    add('token_embedding',emb);add('dur_weight',dw[:,0]);add('dur_bias',db)
    for i in ids:
        base=f'/fs2/encoder/layers.{i}/op/'
        ln1=G.find(base+'layer_norm1/LayerNormalization','LayerNormalization');ln2=G.find(base+'layer_norm2/LayerNormalization','LayerNormalization')
        g1,b1=f32(G.value(ln1.input[1])),f32(G.value(ln1.input[2]));g2,b2=f32(G.value(ln2.input[1])),f32(G.value(ln2.input[2]))
        qkv=G.matmul_weight(G.find(base+'self_attn/in_proj/MatMul','MatMul'));ow=G.matmul_weight(G.find(base+'self_attn/out_proj/MatMul','MatMul'))
        f1n=G.find(base+'ffn/ffn_1/Conv','Conv');f1=f32(G.value(f1n.input[1]));fb1=f32(G.value(f1n.input[2]))
        f2,fb2=G.linear_from_frag(base+'ffn/ffn_2/MatMul',f'fs2.encoder.layers.{i}.op.ffn.ffn_2.bias')
        add(f'l{i}.ln1_gamma',g1);add(f'l{i}.ln1_beta',b1);add(f'l{i}.qkv_weight_m4n16',pack16(qkv));add(f'l{i}.qkv_bias',np.zeros(3*C,np.float32));add(f'l{i}.out_weight_m4n16',pack16(ow));add(f'l{i}.out_bias',np.zeros(C,np.float32));add(f'l{i}.ln2_gamma',g2);add(f'l{i}.ln2_beta',b2);add(f'l{i}.ffn1_weight_m4n16',pack16(conv_flat(f1)));add(f'l{i}.ffn1_bias',fb1);add(f'l{i}.ffn2_weight_m4n16',pack16(f2));add(f'l{i}.ffn2_bias',fb2)
    fln=G.find('/fs2/encoder/layer_norm/LayerNormalization','LayerNormalization');add('final_ln_gamma',G.value(fln.input[1]));add('final_ln_beta',G.value(fln.input[2]))
    stretch_table=None
    try:
        sw1,sb1=G.linear_from_frag('/fs2/stretch_embed/stretch_embed.1/Gemm');sw2,sb2=G.linear_from_frag('/fs2/stretch_embed/stretch_embed.3/Gemm')
    except KeyError:
        # Current deployment exporter may constant-fold the 0..1000 stretch MLP
        # into a lookup table consumed by /fs2/Gather. Preserve that table
        # exactly instead of trying to reconstruct weights that no longer exist.
        gather=G.find('/fs2/Gather','Gather')
        table_name=gather.input[0]
        try: stretch_table=f32(G.value(table_name)).squeeze()
        except KeyError as e:
            raise KeyError(f'stretch MLP was folded but lookup table {table_name!r} is not constant') from e
        if stretch_table.shape!=(1001,C):
            raise ValueError(f'expected folded stretch table [1001,{C}], got {stretch_table.shape}')
        flags|=STRETCH
        sw1=np.zeros((4*C,C),np.float32);sb1=np.zeros(4*C,np.float32)
        sw2=np.zeros((C,4*C),np.float32);sb2=np.zeros(C,np.float32)
    add('stretch_w1_m4n16',pack16(sw1));add('stretch_b1',sb1);add('stretch_w2_m4n16',pack16(sw2));add('stretch_b2',sb2)
    gru=G.find('/fs2/stretch_embed_rnn/GRU','GRU');W=f32(G.value(gru.input[1]))[0];R=f32(G.value(gru.input[2]))[0];B=f32(G.value(gru.input[3]))[0]
    # ONNX GRU gate order z,r,h -> runtime/PyTorch r,z,n.
    def zrh2rzn(x):
        z,r,h=np.split(x,3,axis=0);return np.concatenate([r,z,h],axis=0)
    W=zrh2rzn(W);R=zrh2rzn(R);wb,rb=np.split(B,2);wb=zrh2rzn(wb);rb=zrh2rzn(rb)
    add('gru_weight_ih_l0_m4n16',pack16(W));add('gru_bias_ih_l0',wb);add('gru_weight_hh_l0_m4n16',pack16(R));add('gru_bias_hh_l0',rb)
    pw,pb=G.linear_from_frag('/fs2/pitch_embed/MatMul','fs2.pitch_embed.bias');add('pitch_weight',pw[:,0]);add('pitch_bias',pb)

    scales={'breath':1.0,'voice':1.0,'tension':1.0}
    feats=[('breathiness',BREATH,'breath'),('voicing',VOICE,'voice'),('tension',TENSION,'tension')]
    for inp,bit,short in feats:
        if inp in G.inputs:
            mm=G.find(f'/fs2/{inp}/MatMul','MatMul');w=G.matmul_weight(mm)
            ad=G.find(f'/fs2/{inp}/Add','Add');bi=[z for z in ad.input if z in G.init]
            b=f32(G.value(bi[0])) if bi else np.zeros(C,np.float32)
            mul=G.find(f'/fs2/Mul_{ {"breathiness":4,"voicing":5,"tension":6}[inp] }','Mul');scales[short]=G.scalar(mul.input[1])
            add(short+'_weight',w[:,0]);add(short+'_bias',b);flags|=bit
    gender_consts=[-1.0,1.0,1.0,1.0]
    if 'gender' in G.inputs:
        kw,kb=G.linear_from_frag('/fs2/key_shift_embed/MatMul','fs2.key_shift_embed.bias');add('key_shift_weight',kw[:,0]);add('key_shift_bias',kb);flags|=KEY
        clip=G.find('/fs2/Clip','Clip');gmin,gmax=G.scalar(clip.input[1]),G.scalar(clip.input[2]);m7=G.find('/fs2/Mul_7','Mul');m8=G.find('/fs2/Mul_8','Mul');a=G.scalar(m7.input[1]);b=G.scalar(m8.input[1]);
        if abs(a-b)>1e-7:raise ValueError(f'gender piecewise scales differ ({a},{b}); M25 importer needs extension')
        kscale=G.scalar(G.find('/fs2/Mul_10','Mul').input[1]);gender_consts=[gmin,gmax,a,kscale]
    speed_consts=[0.0,10.0,1.0]
    if 'velocity' in G.inputs:
        # Deployment exporter may deduplicate equal biases.  In the real
        # DongFangZhiZi graph /fs2/speed_embed/Add reuses
        # fs2.key_shift_embed.bias, so never guess a speed_embed.bias name.
        smm=G.find('/fs2/speed_embed/MatMul','MatMul');sw=G.matmul_weight(smm)
        sadd=G.find('/fs2/speed_embed/Add','Add')
        sb_inputs=[z for z in sadd.input if z in G.init or z in G.const]
        if len(sb_inputs)!=1:
            raise KeyError(f'speed Add bias input: {list(sadd.input)} constants={sb_inputs}')
        sb=f32(G.value(sb_inputs[0])).reshape(-1)
        add('speed_weight',sw[:,0]);add('speed_bias',sb);flags|=SPEED
        clip=G.find('/fs2/Clip_1','Clip');speed_consts=[G.scalar(clip.input[1]),G.scalar(clip.input[2]),1.0]
    if 'spk_embed' in G.inputs:flags|=SPK
    if flags&LANG:add('language_embedding',lang)
    # Loader expects optional arrays after base sections, so reorder extras to the end in fixed ABI order.
    # We appended feature arrays while discovering; split base/extras and reconstruct deterministic ordering.
    names={n:x for n,x in arr};base_names=[]
    for n,_ in arr:
        if n in ('language_embedding','language_token_mask','breath_weight','breath_bias','voice_weight','voice_bias','tension_weight','tension_bias','key_shift_weight','key_shift_bias','speed_weight','speed_bias'):continue
        base_names.append(n)
    arr=[(n,names[n]) for n in base_names]
    if flags&LANG:arr.append(('language_embedding',lang))
    if flags&LANGMASK:arr.append(('language_token_mask',lang_mask))
    if flags&BREATH:arr.extend([('breath_weight',names['breath_weight']),('breath_bias',names['breath_bias'])])
    if flags&VOICE:arr.extend([('voicing_weight',names['voice_weight']),('voicing_bias',names['voice_bias'])])
    if flags&TENSION:arr.extend([('tension_weight',names['tension_weight']),('tension_bias',names['tension_bias'])])
    if flags&KEY:arr.extend([('key_shift_weight',names['key_shift_weight']),('key_shift_bias',names['key_shift_bias'])])
    if flags&SPEED:arr.extend([('speed_weight',names['speed_weight']),('speed_bias',names['speed_bias'])])
    arr.append(('feature_constants',np.array(gender_consts+speed_consts,np.float32)))
    if flags&STRETCH:arr.append(('stretch_table',stretch_table))
    # current graph shape shows 2-head non-interleaved RoPE; infer head count via known current exporter.
    H=2;K=3;inter=0;theta=10000.0
    hdr=struct.pack('<8s9I4f4x',b'DSFS25\0\0',1,V,C,len(ids),H,K,inter,nlang,flags,theta,scales['breath'],scales['voice'],scales['tension'])
    secs=write_blob(out/'fs2_acoustic.dsfs',hdr,arr)
    return {'magic':'DSFS25','vocab_size':V,'hidden_size':C,'num_layers':len(ids),'num_heads':H,'num_languages':nlang,'feature_flags':flags,'cross_lingual_token_idx':cross_ids,'feature_constants':gender_consts+speed_consts,'scales':scales,'sections':secs}

def pack_aux(G,out):
    inn=G.find('/aux_decoder/decoder/inconv/Conv','Conv');inw=f32(G.value(inn.input[1]));inb=f32(G.value(inn.input[2]));C,I,K=inw.shape
    outn=G.find('/aux_decoder/decoder/outconv/Conv','Conv');outw=f32(G.value(outn.input[1]));outb=f32(G.value(outn.input[2]));D,C2,K2=outw.shape
    ids=sorted({int(m.group(1)) for n in G.nodes for m in [re.search(r'/aux_decoder/decoder/conv\.(\d+)/dwconv/Conv$',n.name)] if m});
    if ids!=list(range(len(ids))) or K!=7 or K2!=7 or C2!=C:raise ValueError(('aux',ids,inw.shape,outw.shape))
    arr=[('in_weight_m4n16',pack16(conv_flat(inw))),('in_bias',inb)]
    for i in ids:
        b=f'/aux_decoder/decoder/conv.{i}/';dw=G.find(b+'dwconv/Conv','Conv');ln=G.find(b+'norm/LayerNormalization','LayerNormalization');w1,b1=G.linear_from_frag(b+'pwconv1/MatMul',f'aux_decoder.decoder.conv.{i}.pwconv1.bias');w2,b2=G.linear_from_frag(b+'pwconv2/MatMul',f'aux_decoder.decoder.conv.{i}.pwconv2.bias');mul=G.find(b+'Mul','Mul');gamma_in=[z for z in mul.input if z in G.init and 'gamma' in z]
        if len(gamma_in)!=1:raise KeyError(f'aux gamma {i}: {gamma_in}')
        dww=f32(G.value(dw.input[1]))[:,0,:]
        arr.extend([(f'b{i}.dw_weight_tap_major',dww.T),(f'b{i}.dw_bias',G.value(dw.input[2])),(f'b{i}.ln_gamma',G.value(ln.input[1])),(f'b{i}.ln_beta',G.value(ln.input[2])),(f'b{i}.pw1_weight_m4n16',pack16(w1)),(f'b{i}.pw1_bias',b1),(f'b{i}.pw2_weight_m4n16',pack16(w2)),(f'b{i}.pw2_bias',b2),(f'b{i}.gamma',G.value(gamma_in[0]))])
    arr.extend([('out_weight_m4n16',pack16(conv_flat(outw))),('out_bias',outb)])
    hdr=struct.pack('<8s8I24x',b'DSAUX20\0',1,I,C,D,len(ids),7,4,16);secs=write_blob(out/'aux_convnext.dsa',hdr,arr)
    return {'magic':'DSAUX20','input_dim':I,'channels':C,'output_dim':D,'num_layers':len(ids),'sections':secs}

def lin_bias(G,suffix):return G.linear_from_bias(suffix)
def pack_rf(G,out):
    try: iw,ib=G.linear_by_node('/diffusion/velocity_fn/input_projection/','diffusion.velocity_fn.input_projection.bias')
    except KeyError: iw,ib=lin_bias(G,'diffusion.velocity_fn.input_projection.bias')
    cws=[k for k in G.init if k.endswith('diffusion.velocity_fn.conditioner_projection.weight')]
    cbs=[k for k in G.init if k.endswith('diffusion.velocity_fn.conditioner_projection.bias')]
    if len(cws)!=1 or len(cbs)!=1:raise KeyError(('conditioner_projection',cws,cbs))
    cw=f32(G.value(cws[0]));cb=f32(G.value(cbs[0]))
    if cw.ndim==3:cw=cw[:,:,0]
    try: tw1,tb1=G.linear_by_node('/diffusion/velocity_fn/diffusion_embedding/1/','diffusion.velocity_fn.diffusion_embedding.1.bias')
    except KeyError: tw1,tb1=lin_bias(G,'diffusion.velocity_fn.diffusion_embedding.1.bias')
    try: tw2,tb2=G.linear_by_node('/diffusion/velocity_fn/diffusion_embedding/3/','diffusion.velocity_fn.diffusion_embedding.3.bias')
    except KeyError: tw2,tb2=lin_bias(G,'diffusion.velocity_fn.diffusion_embedding.3.bias')
    try: ow,ob=G.linear_by_node('/diffusion/velocity_fn/output_projection/','diffusion.velocity_fn.output_projection.bias')
    except KeyError: ow,ob=lin_bias(G,'diffusion.velocity_fn.output_projection.bias')
    normw=[k for k in G.init if k.endswith('diffusion.velocity_fn.norm.weight')];normb=[k for k in G.init if k.endswith('diffusion.velocity_fn.norm.bias')]
    if len(normw)!=1 or len(normb)!=1:raise KeyError(('rf norm',normw,normb))
    pg,pb=f32(G.value(normw[0])),f32(G.value(normb[0]));C,I=iw.shape;C2,Q=cw.shape
    ids=sorted({int(m.group(1)) for n in G.nodes for m in [re.search(r'residual_layers\.(\d+)',n.name)] if m})
    if ids!=list(range(len(ids))) or not ids:raise ValueError(f'RF layers {ids}')
    blocks=[];Hid=None
    for i in ids:
        # named parameters survive for LN/Conv; linear weights may be anonymous so recover by bias.
        def key(s):
            q=[k for k in G.init if k.endswith(f'diffusion.velocity_fn.residual_layers.{i}.net.{s}')]
            if len(q)!=1:
                raise KeyError((i,s,q[:8]))
            return q[0]
        g=f32(G.value(key('0.weight')));bb=f32(G.value(key('0.bias')));dw=f32(G.value(key('2.weight')));dwb=f32(G.value(key('2.bias')))
        try:w1,b1=G.linear_by_node(f'/diffusion/velocity_fn/residual_layers.{i}/net.4/',f'diffusion.velocity_fn.residual_layers.{i}.net.4.bias')
        except KeyError:w1,b1=lin_bias(G,f'diffusion.velocity_fn.residual_layers.{i}.net.4.bias')
        try:w2,b2=G.linear_by_node(f'/diffusion/velocity_fn/residual_layers.{i}/net.6/',f'diffusion.velocity_fn.residual_layers.{i}.net.6.bias')
        except KeyError:w2,b2=lin_bias(G,f'diffusion.velocity_fn.residual_layers.{i}.net.6.bias')
        try:w3,b3=G.linear_by_node(f'/diffusion/velocity_fn/residual_layers.{i}/net.8/',f'diffusion.velocity_fn.residual_layers.{i}.net.8.bias')
        except KeyError:w3,b3=lin_bias(G,f'diffusion.velocity_fn.residual_layers.{i}.net.8.bias')
        if dw.ndim==3:dw=dw[:,0,:]
        h=w1.shape[0]//2;Hid=h if Hid is None else Hid
        if h!=Hid:raise ValueError('RF hidden mismatch')
        blocks.append((g,bb,dw,dwb,w1,b1,w2,b2,w3,b3))
    has_atan=any(n.op_type=='Atan' for n in G.nodes);has_soft=any(n.op_type.lower()=='softsign' for n in G.nodes)
    glu='atan' if has_atan else 'softsign'
    if not has_atan and not has_soft:print('  warning: no explicit Atan/Softsign op found recursively; assuming SoftSignGLU')
    arr=[('input_weight_m4n16',pack16(iw)),('input_bias',ib),('condition_weight_m4n16',pack16(cw)),('condition_bias',cb),('time1_weight_m4n16',pack16(tw1)),('time1_bias',tb1),('time2_weight_m4n16',pack16(tw2)),('time2_bias',tb2)]
    for i,(g,bb,dw,dwb,w1,b1,w2,b2,w3,b3) in enumerate(blocks):
        arr.extend([(f'b{i}.ln_gamma',g),(f'b{i}.ln_beta',bb),(f'b{i}.dw_weight_tap_major',dw.T),(f'b{i}.dw_bias',dwb),(f'b{i}.glu1_weight',pack16(w1)),(f'b{i}.glu1_bias',b1),(f'b{i}.glu2_weight',pack16(w2)),(f'b{i}.glu2_bias',b2),(f'b{i}.out_weight_m4n16',pack16(w3)),(f'b{i}.out_bias',b3)])
    arr.extend([('post_norm_gamma',pg),('post_norm_beta',pb),('output_weight_m4n16',pack16(ow)),('output_bias',ob)])
    glu_code=1 if glu=='atan' else 2
    hdr=struct.pack('<8s12I8x',b'DSLYNX7\0',1,I,Q,C,Hid,len(ids),31,glu_code,4,16,4,8);secs=write_blob(out/'lynxnet2.dsn',hdr,arr)
    return {'magic':'DSLYNX7','input_dim':I,'condition_dim':Q,'channels':C,'hidden_dim':Hid,'num_layers':len(ids),'glu':glu,'sections':secs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('onnx',type=Path);ap.add_argument('--model-dir',type=Path);ap.add_argument('--out',type=Path,default=Path('packed_onnx_m25'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m=onnx.load(a.onnx,load_external_data=True);G=Graph(m);print(f'M25 ONNX: nodes(recursive)={len(G.nodes)} initializers={len(G.init)} inputs={sorted(G.inputs)}')
    try:
        fs=pack_fs2(G,a.out);print('  packed FS2')
        au=pack_aux(G,a.out);print('  packed AUX')
        rf=pack_rf(G,a.out);print('  packed RF')
    except Exception as e:
        print(f'M25 ONNX importer failed: {type(e).__name__}: {e}',flush=True)
        raise
    # spec range from exported aux denorm y = norm*k+b
    mul=G.find('/aux_decoder/Mul','Mul');add=G.find('/aux_decoder/Add','Add');k=G.value(mul.input[1]);b=G.value([z for z in add.input if z!=mul.output[0]][0]);lo=f32(b-k).reshape(-1);hi=f32(b+k).reshape(-1)
    lo.tofile(a.out/'spec_min.f32');hi.tofile(a.out/'spec_max.f32')
    csv=lambda x:','.join(f'{float(v):.9g}' for v in x)
    conf=['# DiffSinger-ASM M25 ONNX deployment model',f'mel_bins={rf["input_dim"]}',f'hidden_size={fs["hidden_size"]}',f'feature_flags={fs["feature_flags"]}',f'num_languages={fs["num_languages"]}',f'spec_range_dims={lo.size}',f'spec_min={csv(lo)}',f'spec_max={csv(hi)}','spec_min_file=spec_min.f32','spec_max_file=spec_max.f32','t_start=0.4','time_scale_factor=1000','steps=20','sample_rate=44100','hop_size=512','deployment_depth_semantics=t_start=max(1-depth,0)']
    (a.out/'model.conf').write_text('\n'.join(conf)+'\n',encoding='utf8')
    summary={'format':'DiffSinger-ASM M25 ONNX deployment','source_onnx':str(a.onnx),'graph_inputs':sorted(G.inputs),'fs2':fs,'aux':au,'rf':rf,'spec_min':lo.tolist(),'spec_max':hi.tolist()}
    (a.out/'model.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    if a.model_dir:
        for fn in ('phonemes.json','languages.json','dsconfig.yaml','character.yaml','character.txt'):
            p=a.model_dir/fn
            if p.exists():shutil.copy2(p,a.out/fn)
        for p in a.model_dir.glob('*.emb'):shutil.copy2(p,a.out/p.name)
    print(f'packed M25 ONNX deployment -> {a.out}')
    print(f'  FS2 V={fs["vocab_size"]} C={fs["hidden_size"]} L={fs["num_layers"]} langs={fs["num_languages"]} flags=0x{fs["feature_flags"]:x}')
    if fs.get('cross_lingual_token_idx'):
        print(f'  cross-lingual token ids ({len(fs["cross_lingual_token_idx"])}): {fs["cross_lingual_token_idx"][:32]}' + (' ...' if len(fs["cross_lingual_token_idx"])>32 else ''))
    print(f'  AUX {au["input_dim"]}->{au["channels"]}x{au["num_layers"]}->{au["output_dim"]}')
    print(f'  RF I={rf["input_dim"]} Q={rf["condition_dim"]} C={rf["channels"]} H={rf["hidden_dim"]} L={rf["num_layers"]} glu={rf["glu"]}')
if __name__=='__main__':main()
