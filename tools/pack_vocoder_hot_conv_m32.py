#!/usr/bin/env python3
import argparse, json, struct, pathlib
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

MAGIC=b'DSVOC32\0'

def attrs(n):
    d={}
    for a in n.attribute:
        if a.type==onnx.AttributeProto.INT: d[a.name]=int(a.i)
        elif a.type==onnx.AttributeProto.INTS: d[a.name]=list(a.ints)
    return d

def pack4(w):
    co,ci,k=w.shape
    if co%4: raise ValueError(f'Cout must be multiple of 4, got {co}')
    return w.reshape(co//4,4,ci,k).transpose(0,2,3,1).copy()

def select_node(g, frag):
    convs=[n for n in g.node if n.op_type=='Conv']
    if frag:
        q=[n for n in convs if n.name==frag]
        if not q:q=[n for n in convs if frag in n.name]
        if len(q)!=1: raise RuntimeError(f'Conv selector {frag!r} matched {len(q)} nodes')
        return q[0]
    # Default to the hottest node observed in M31; fall back to largest k>=7 conv.
    for name in ['/generator/resblocks.5/convs1.0/Conv','/generator/resblocks.5/convs2.0/Conv']:
        q=[n for n in convs if n.name==name]
        if q:return q[0]
    init={x.name:numpy_helper.to_array(x) for x in g.initializer}
    cand=[]
    for n in convs:
        if len(n.input)>1 and n.input[1] in init and init[n.input[1]].ndim==3:
            w=init[n.input[1]]; cand.append((w.size,n))
    if not cand:raise RuntimeError('no Conv candidates')
    return max(cand,key=lambda z:z[0])[1]

def add_output(g,name):
    if any(o.name==name for o in g.output):return
    g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,None))

def main():
    ap=argparse.ArgumentParser(description='Extract one real NSF-HiFiGAN Conv as a pure-ASM M32 benchmark bundle.')
    ap.add_argument('onnx');ap.add_argument('--out',required=True);ap.add_argument('--node',default=None)
    ap.add_argument('--frames',type=int,default=48);ap.add_argument('--f0',type=float,default=220.0)
    a=ap.parse_args(); out=pathlib.Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    m=onnx.load(a.onnx); g=m.graph; init={x.name:numpy_helper.to_array(x) for x in g.initializer}
    n=select_node(g,a.node); at=attrs(n)
    if len(n.input)<2 or n.input[1] not in init: raise RuntimeError('Conv weight is not an initializer')
    w=np.asarray(init[n.input[1]],dtype=np.float32)
    if w.ndim!=3: raise RuntimeError(f'expected Conv1d weight, got {w.shape}')
    co,ci,k=w.shape
    b=np.asarray(init[n.input[2]],dtype=np.float32) if len(n.input)>2 and n.input[2] in init else np.zeros(co,np.float32)
    strides=at.get('strides',[1]); dil=at.get('dilations',[1]); pads=at.get('pads',[0,0]); group=at.get('group',1)
    if strides!=[1] or group!=1: raise RuntimeError(f'M32 supports stride=1 group=1 only; node has stride={strides} group={group}')
    if len(dil)!=1:raise RuntimeError(f'bad dilation {dil}')
    if len(pads)!=2 or pads[0]!=pads[1]:raise RuntimeError(f'M32 currently requires symmetric Conv1d padding, got {pads}')
    add_output(g,n.input[0]); add_output(g,n.output[0])
    patched=out.with_suffix('.probe.onnx'); onnx.save(m,patched)
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession(str(patched),so,providers=['CPUExecutionProvider'])
    feeds={}
    for inp in sess.get_inputs():
        shape=inp.shape
        if inp.name=='mel': feeds[inp.name]=np.zeros((1,a.frames,128),np.float32)
        elif inp.name=='f0': feeds[inp.name]=np.full((1,a.frames),a.f0,np.float32)
        else: raise RuntimeError(f'unexpected vocoder input {inp.name} {shape}')
    xin,yref=sess.run([n.input[0],n.output[0]],feeds)
    if xin.ndim!=3 or yref.ndim!=3 or xin.shape[0]!=1 or yref.shape[0]!=1:raise RuntimeError(f'unexpected tensors {xin.shape} -> {yref.shape}')
    # Vocoder Conv is NCT.
    x=np.asarray(xin[0],dtype=np.float32); y=np.asarray(yref[0],dtype=np.float32)
    if x.shape[0]!=ci or y.shape[0]!=co:raise RuntimeError(f'channel mismatch w={w.shape} x={x.shape} y={y.shape}')
    tin=x.shape[1]; tout=y.shape[1]; pad=int(pads[0]); dilation=int(dil[0]); wp=pack4(w)
    hdr=struct.pack('<8s10I16x',MAGIC,1,ci,co,k,tin,tout,pad,dilation,1,group)
    with open(out,'wb') as f:
        f.write(hdr);f.write(b.tobytes());f.write(wp.tobytes());f.write(x.tobytes());f.write(y.tobytes())
    meta={'node':n.name,'input_tensor':n.input[0],'output_tensor':n.output[0],'weight_shape':list(w.shape),'input_shape':list(xin.shape),'output_shape':list(yref.shape),'pad':pad,'dilation':dilation,'stride':1,'group':group,'macs':int(co*tout*ci*k)}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    try:patched.unlink()
    except OSError:pass
    print('M32 real hot Conv packed:',json.dumps(meta,ensure_ascii=False))
    print('bundle:',out)
if __name__=='__main__':main()
