#!/usr/bin/env python3
import argparse, copy, json, math
from pathlib import Path
import numpy as np
from dsv35_common import Builder, OPS

def attr_dict(node, onnx):
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

def main():
    ap=argparse.ArgumentParser(description='Compile fixed-shape NSF-HiFiGAN ONNX to DSVOC35 pure-native bundle')
    ap.add_argument('onnx');ap.add_argument('--frames',type=int,default=48);ap.add_argument('--out');ap.add_argument('--work');ap.add_argument('--preflight',action='store_true');ap.add_argument('--seed',type=int,default=35);ap.add_argument('--enable-residual-fusion',action='store_true',help='M38 experiment; disabled by default in M39 because target-machine A/B regressed');ap.add_argument('--residual-scope',choices=['none','full64','k7ge128','all3711'],default='none',help='residual fusion scope; all3711 enables every aligned K3/K7/K11 residual stage including 2-D late stages');ap.add_argument('--vnni-scope',choices=['none','stage128','all-k711'],default='stage128',help='M40: embed prequantized AVX-VNNI weights; runtime DSASM_VNNI selects whether to execute them')
    a=ap.parse_args()
    if not a.preflight and (not a.out or not a.work):ap.error('--out and --work are required unless --preflight is used')
    import onnx, onnxruntime as ort
    from onnx import numpy_helper, helper, TensorProto
    m=onnx.load(a.onnx);g=m.graph; init={x.name:numpy_helper.to_array(x) for x in g.initializer}
    inputs={x.name:x for x in g.input if x.name not in init}
    if 'mel' not in inputs or 'f0' not in inputs: raise SystemExit(f'expected mel/f0 graph inputs, got {list(inputs)}')
    rng=np.random.default_rng(a.seed)
    # Realistic range is not required for shapes, but keeps activation magnitudes sane.
    mel=(rng.standard_normal((1,a.frames,128),dtype=np.float32)*0.8-4.0).astype(np.float32)
    tt=np.arange(a.frames,dtype=np.float32);f0=(220.0+8.0*np.sin(2*np.pi*tt/max(1,a.frames))).reshape(1,-1).astype(np.float32)
    feed={'mel':mel,'f0':f0}
    # Add every intermediate as a debug output. ORT determines exact static shapes
    # after the frame count is fixed; this avoids reimplementing ONNX shape inference.
    dbg=copy.deepcopy(m);existing={o.name for o in dbg.graph.output}
    for n in dbg.graph.node:
        for name in n.output:
            if name and name not in existing:
                dbg.graph.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,None));existing.add(name)
    so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if a.preflight:
        sess=ort.InferenceSession(dbg.SerializeToString(),so,providers=['CPUExecutionProvider'])
        w=None
    else:
        w=Path(a.work);w.mkdir(parents=True,exist_ok=True);dbg_path=w/'m35_shape_debug.onnx';onnx.save(dbg,dbg_path)
        sess=ort.InferenceSession(str(dbg_path),so,providers=['CPUExecutionProvider'])
    out_names=[o.name for o in sess.get_outputs()]; vals=sess.run(out_names,feed); captured=dict(zip(out_names,vals))
    # Original golden is taken from the same unoptimized math graph; ORT is offline only.
    wave=np.asarray(captured.get('waveform',vals[0]),dtype=np.float32).reshape(-1)
    if w is not None:
        mel.reshape(-1).tofile(w/'mel.f32');f0.reshape(-1).tofile(w/'f0.f32');wave.tofile(w/'golden_wave.f32')
    shapes={'mel':mel.shape,'f0':f0.shape};shapes.update({k:np.asarray(v).shape for k,v in captured.items()})
    b=Builder(a.frames,128,wave.size); ids={}
    ids['mel']=b.add_work(mel.shape);ids['f0']=b.add_work(f0.shape)
    const_cache={}
    def const_id(name):
        if name in const_cache:return const_cache[name]
        if name not in init:raise KeyError(f'initializer {name!r} not found')
        arr=np.asarray(init[name])
        if arr.dtype.kind not in 'fc':raise TypeError(f'{name}: expected float initializer, got {arr.dtype}')
        tid=b.add_const(arr.astype(np.float32));const_cache[name]=tid;ids[name]=tid;return tid
    def input_id(name):
        if name in ids:return ids[name]
        if name in init:return const_id(name)
        raise KeyError(f'tensor id not available for {name!r}')
    def out_id(name, alloc=None):
        if name in ids:
            if alloc:b.grow_work(ids[name],alloc)
            return ids[name]
        if name not in shapes:raise KeyError(f'no captured shape for {name}')
        tid=b.add_work(shapes[name],alloc);ids[name]=tid;return tid
    def resolve_i64(name, default=None):
        if not name:return np.asarray(default,dtype=np.int64)
        if name not in init:raise KeyError(f'{name}: dynamic integer control tensor unsupported in M35')
        return np.asarray(init[name],dtype=np.int64).reshape(-1)
    producers={}
    consumers={}
    for ni,n in enumerate(g.node):
        for o in n.output:
            if o: producers[o]=ni
        for x in n.input:
            if x: consumers.setdefault(x,[]).append(ni)
    fuse_leaky={}
    fuse_residual={}
    skip_nodes=set()
    # M49 keeps legacy M38 all-residual fusion for compatibility, but adds a
    # selective production experiment: only full/channel-owner K3/K7/K11
    # shapes with Cout>=64 and T divisible by 24.  C32/C16 stay on the proven
    # 2-D range path, where M47 showed t24 increases cycles despite fewer insns.
    if a.enable_residual_fusion or a.residual_scope!='none':
      for ni,n in enumerate(g.node):
        if n.op_type!='Conv' or not n.output or n.input[1] not in init: continue
        W=np.asarray(init[n.input[1]])
        if W.ndim!=3 or int(W.shape[0])<8 or int(W.shape[0])%8: continue
        if a.residual_scope in ('full64','k7ge128','all3711'):
            Cout,Cin,K=map(int,W.shape)
            if a.residual_scope=='full64' and (Cout<64 or K not in (3,7,11)): continue
            if a.residual_scope=='k7ge128' and (Cout<128 or K!=7): continue
            if a.residual_scope=='all3711' and K not in (3,7,11): continue
            if n.output[0] not in shapes or int(shapes[n.output[0]][-1])%24: continue
        cns=consumers.get(n.output[0],[])
        if len(cns)!=1: continue
        ai=cns[0];an=g.node[ai]
        if an.op_type!='Add' or len(an.input)<2 or not an.output: continue
        if an.input[0]==n.output[0]: res=an.input[1]
        elif an.input[1]==n.output[0]: res=an.input[0]
        else: continue
        if n.output[0] not in shapes or res not in shapes or an.output[0] not in shapes: continue
        if tuple(shapes[n.output[0]])!=tuple(shapes[res]) or tuple(shapes[n.output[0]])!=tuple(shapes[an.output[0]]): continue
        fuse_residual[ni]=(res,an.output[0],ai)
        skip_nodes.add(ai)
    for ni,n in enumerate(g.node):
        if n.op_type!='Conv' or not n.input: continue
        pi=producers.get(n.input[0])
        if pi is None: continue
        pn=g.node[pi]
        if pn.op_type=='LeakyRelu' and len(consumers.get(n.input[0],[]))==1:
            pad=attr_dict(pn,onnx)
            fuse_leaky[ni]=(pn.input[0],float(pad.get('alpha',0.01)))
            skip_nodes.add(pi)
    unsupported=[(ni,n.op_type,n.name) for ni,n in enumerate(g.node) if n.op_type not in OPS]
    if unsupported:
        raise SystemExit('unsupported ONNX ops in M35.1 preflight: '+repr(unsupported[:40]))
    convs=cts=0; fused_leaky_count=0; fused_residual_count=0; pack8_count=0; conv2d_candidates=0; vnni_candidates=0; conv_shapes={}
    for ni,n in enumerate(g.node):
        if ni in skip_nodes:
            continue
        typ=n.op_type
        ad=attr_dict(n,onnx); raw_out=n.output[0]
        out=fuse_residual[ni][1] if typ=='Conv' and ni in fuse_residual else raw_out
        oid=out_id(out)
        if typ in ('Add','Sub','Mul','Div'):
            b.add_op(typ,[input_id(n.input[0]),input_id(n.input[1])],oid)
        elif typ=='Mod':
            if int(ad.get('fmod',0))!=1: raise ValueError(f'{n.name}: only floating fmod=1 supported')
            b.add_op(typ,[input_id(n.input[0]),input_id(n.input[1])],oid)
        elif typ=='LeakyRelu': b.add_op(typ,[input_id(n.input[0])],oid,f=[float(ad.get('alpha',0.01))])
        elif typ in ('Tanh','Sin'): b.add_op(typ,[input_id(n.input[0])],oid)
        elif typ=='CumSum':
            axis=int(resolve_i64(n.input[1])[0]);b.add_op(typ,[input_id(n.input[0])],oid,p=[axis])
        elif typ in ('Reshape','Squeeze','Unsqueeze'):
            # Fixed-shape AOT: these are pure shape/view operations. The runtime
            # stores tensors densely, so preserving flat element order is sufficient.
            b.add_op(typ,[input_id(n.input[0])],oid)
        elif typ=='Transpose':
            perm=list(ad.get('perm',list(reversed(range(len(shapes[n.input[0]]))))));b.add_op(typ,[input_id(n.input[0])],oid,p=perm)
        elif typ=='Slice':
            xshape=tuple(int(x) for x in shapes[n.input[0]]);rank=len(xshape)
            starts=resolve_i64(n.input[1]);ends=resolve_i64(n.input[2]);axes=resolve_i64(n.input[3],np.arange(len(starts)));steps=resolve_i64(n.input[4],np.ones(len(starts),np.int64)) if len(n.input)>4 and n.input[4] else np.ones(len(starts),np.int64)
            ss=[0]*rank;st=[1]*rank
            for s,e,ax,step in zip(starts,ends,axes,steps):
                ax=int(ax);ax=ax+rank if ax<0 else ax; s0,_,sp=slice(int(s),int(e),int(step)).indices(xshape[ax]);ss[ax]=s0;st[ax]=sp
            b.add_op(typ,[input_id(n.input[0])],oid,p=ss+[0]*(4-rank)+st+[1]*(4-rank))
        elif typ=='Pad':
            if ad.get('mode',b'constant') not in (b'constant','constant'):raise ValueError(f'{n.name}: only constant Pad supported')
            pads=resolve_i64(n.input[1] if len(n.input)>1 else '',ad.get('pads'));rank=len(shapes[n.input[0]]);beg=[int(x) for x in pads[:rank]]
            cv=0.0
            if len(n.input)>2 and n.input[2]:cv=float(np.asarray(init[n.input[2]]).reshape(-1)[0])
            b.add_op(typ,[input_id(n.input[0])],oid,p=beg,f=[cv])
        elif typ=='Conv':
            xshape=tuple(int(x) for x in shapes[n.input[0]]);oshape=tuple(int(x) for x in shapes[out]);
            if len(xshape)!=3 or xshape[0]!=1 or len(oshape)!=3 or oshape[0]!=1:raise ValueError(f'{n.name}: only N=1 Conv1d supported {xshape}->{oshape}')
            W=np.asarray(init[n.input[1]],dtype=np.float32);Cout,Cin,K=map(int,W.shape);group=int(ad.get('group',1));strides=list(ad.get('strides',[1]));dils=list(ad.get('dilations',[1]));pads=list(ad.get('pads',[0,0]));
            if group!=1 or strides!=[1] or len(pads)!=2 or pads[0]!=pads[1]:raise ValueError(f'{n.name}: unsupported Conv attrs group={group} stride={strides} pads={pads}')
            pack=8 if Cout>=8 and (Cout%8)==0 else 4
            if pack==8 and (Cout//8)<8 and int(oshape[2])>=1024: conv2d_candidates+=1
            sk=f"{Cin}x{Cout}xK{K}@T{int(oshape[2])}d{int(dils[0])}"
            conv_shapes[sk]=conv_shapes.get(sk,0)+1
            Cpad=(Cout+pack-1)//pack*pack; packed=np.zeros((Cpad//pack,Cin,K,pack),np.float32)
            for oc in range(Cout):packed[oc//pack,:,:,oc%pack]=W[oc]
            bias=np.zeros(Cpad,np.float32)
            if len(n.input)>2 and n.input[2]:bias[:Cout]=np.asarray(init[n.input[2]],dtype=np.float32).reshape(-1)
            wid=b.add_const(packed);bid=b.add_const(bias);b.grow_work(oid,Cpad*int(oshape[2]))
            flags=0;alpha=0.0;xname=n.input[0];reserved=0
            # M40: selectively embed per-output-channel S8 weights for the
            # C=128,T=3072 K7/K11 stage that M39.1 proved worthwhile. The
            # FP32 packed weights remain in the bundle for exact fallback/A-B.
            eligible_vnni = (pack==8 and K in (7,11))
            if a.vnni_scope=='stage128': eligible_vnni = eligible_vnni and Cin==128 and Cout==128 and int(oshape[2])==3072
            elif a.vnni_scope=='all-k711': eligible_vnni = eligible_vnni
            else: eligible_vnni = False
            if eligible_vnni:
                kred=Cin*K; k4=(kred+3)//4
                ws=np.empty(Cout,np.float32); qw=np.zeros_like(W,dtype=np.int8); corr=np.empty(Cout,np.int32)
                for oc in range(Cout):
                    mx=float(np.max(np.abs(W[oc]))); sc=mx/127.0 if mx>0 else 1.0; ws[oc]=sc
                    q=np.rint(W[oc]/sc).clip(-127,127).astype(np.int8); qw[oc]=q; corr[oc]=-128*int(q.astype(np.int32).sum())
                wp=np.zeros((Cout//8,k4,8,4),np.int8)
                for oc in range(Cout):
                    flat=qw[oc].reshape(-1)
                    for rr,v in enumerate(flat): wp[oc//8,rr//4,oc%8,rr%4]=v
                blob=ws.tobytes()+corr.tobytes()+wp.tobytes()
                reserved=b.add_blob(blob); flags|=4; vnni_candidates+=1
            if ni in fuse_leaky:
                xname,alpha=fuse_leaky[ni];flags|=1;fused_leaky_count+=1
                xshape=tuple(int(x) for x in shapes[xname])
            residual_id=0
            if ni in fuse_residual:
                resname,_,_=fuse_residual[ni];residual_id=input_id(resname);flags|=2|8;fused_residual_count+=1
            if pack==8: pack8_count+=1
            b.add_op(typ,[input_id(xname),wid,bid],oid,p=[Cin,Cout,Cpad,K,int(xshape[2]),int(pads[0]),int(dils[0]),pack,residual_id],f=[alpha],flags=flags,reserved=reserved);convs+=1
        elif typ=='ConvTranspose':
            xshape=tuple(int(x) for x in shapes[n.input[0]]);oshape=tuple(int(x) for x in shapes[out]);W=np.asarray(init[n.input[1]],dtype=np.float32)
            if len(xshape)!=3 or xshape[0]!=1 or len(oshape)!=3 or oshape[0]!=1 or W.ndim!=3:raise ValueError(f'{n.name}: only N=1 ConvTranspose1d')
            Cin,Cout,K=map(int,W.shape);group=int(ad.get('group',1));strides=list(ad.get('strides',[1]));dils=list(ad.get('dilations',[1]));pads=list(ad.get('pads',[0,0]));opad=list(ad.get('output_padding',[0]));
            if group!=1 or dils!=[1] or len(strides)!=1 or len(pads)!=2 or pads[0]!=pads[1] or opad not in ([0],[]):raise ValueError(f'{n.name}: unsupported ConvTranspose attrs')
            wom=np.transpose(W,(1,0,2)).copy();bias=np.zeros(Cout,np.float32)
            if len(n.input)>2 and n.input[2]:bias[:]=np.asarray(init[n.input[2]],dtype=np.float32).reshape(-1)
            wid=b.add_const(wom);bid=b.add_const(bias);b.add_op(typ,[input_id(n.input[0]),wid,bid],oid,p=[Cin,Cout,K,int(xshape[2]),int(pads[0]),int(strides[0])]);cts+=1
    if 'waveform' not in ids:raise SystemExit(f'waveform tensor not found, outputs={list(ids)[-20:]}')
    naive_arena,planned_arena=b.plan_arena_lifetimes(ids['mel'],ids['f0'],ids['waveform'])
    meta={'format':'DSVOC35','frames':a.frames,'mel_bins':128,'samples':wave.size,'nodes':len(g.node),'ops':len(b.ops),'tensors':len(b.tensors),'arena_floats':b.arena,'arena_mib':b.arena*4/2**20,'arena_naive_mib':naive_arena*4/2**20,'arena_reduction_x':(naive_arena/planned_arena if planned_arena else 1.0),'const_mib':len(b.const)/2**20,'conv':convs,'convtranspose':cts,'fused_leaky_conv':fused_leaky_count,'fused_residual_add':fused_residual_count,'pack8_conv':pack8_count,'conv2d_candidates_8w':conv2d_candidates,'vnni_candidates':vnni_candidates,'vnni_scope':a.vnni_scope,'conv_shapes':conv_shapes,'supported_ops':sorted(OPS),'revision':'M40-selective-vnni-stage128'+(('-residual-'+a.residual_scope) if a.residual_scope!='none' else ('-residual-experimental' if a.enable_residual_fusion else ''))}
    if a.preflight:
        print(json.dumps({'preflight':'ok',**meta},sort_keys=True));return
    outp=b.write(a.out,ids['mel'],ids['f0'],ids['waveform'])
    Path(str(Path(a.out).with_suffix('.json'))).write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('M35 packed pure-native vocoder:',json.dumps(meta))
    print('bundle:',outp);print('sample mel/f0/golden:',w/'mel.f32',w/'f0.f32',w/'golden_wave.f32')
if __name__=='__main__':main()
