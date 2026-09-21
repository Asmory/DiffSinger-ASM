#!/usr/bin/env python3
import argparse,json,pathlib,struct
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
import onnxruntime as ort
MAGIC=b'DSVCT33\0'

def attrs(n):
    d={}
    for a in n.attribute:
        if a.type==onnx.AttributeProto.INT:d[a.name]=int(a.i)
        elif a.type==onnx.AttributeProto.INTS:d[a.name]=list(a.ints)
    return d

def pack_output_major(w):
    return np.asarray(w,np.float32).transpose(1,0,2).copy()

def add_output(g,name):
    if not any(o.name==name for o in g.output):g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('onnx');ap.add_argument('--node',default='/generator/ups.1/ConvTranspose');ap.add_argument('--frames',type=int,default=48);ap.add_argument('--f0',type=float,default=220.);ap.add_argument('--out',required=True);a=ap.parse_args();m=onnx.load(a.onnx);g=m.graph;init={x.name:numpy_helper.to_array(x) for x in g.initializer}
    q=[n for n in g.node if n.name==a.node] or [n for n in g.node if n.op_type=='ConvTranspose' and a.node in n.name]
    if len(q)!=1:
        raise RuntimeError(f'ConvTranspose selector matched {len(q)}')
    n=q[0]
    if n.input[1] not in init:raise RuntimeError('weight not initializer')
    w=np.asarray(init[n.input[1]],np.float32);ci,co,k=w.shape;b=np.asarray(init[n.input[2]],np.float32) if len(n.input)>2 and n.input[2] in init else np.zeros(co,np.float32);at=attrs(n);pads=at.get('pads',[0,0]);dil=at.get('dilations',[1]);stride=at.get('strides',[1]);group=at.get('group',1);op=at.get('output_padding',[0])
    if w.ndim!=3 or len(pads)!=2 or pads[0]!=pads[1] or len(dil)!=1 or dil!=[1] or len(stride)!=1 or group!=1 or op not in ([0],[]):raise RuntimeError(f'unsupported ConvTranspose w={w.shape} pads={pads} dil={dil} stride={stride} group={group} output_padding={op}')
    add_output(g,n.input[0]);add_output(g,n.output[0]);out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);probe=out.with_suffix('.probe.onnx');onnx.save(m,probe);so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;s=ort.InferenceSession(str(probe),so,providers=['CPUExecutionProvider']);feeds={}
    for i in s.get_inputs():
        if i.name=='mel':feeds[i.name]=np.zeros((1,a.frames,128),np.float32)
        elif i.name=='f0':feeds[i.name]=np.full((1,a.frames),a.f0,np.float32)
        else:raise RuntimeError(f'unexpected input {i.name}')
    xin,yref=s.run([n.input[0],n.output[0]],feeds);x=np.asarray(xin[0],np.float32);y=np.asarray(yref[0],np.float32)
    if x.shape[0]!=ci or y.shape[0]!=co:raise RuntimeError(f'channel mismatch {x.shape}->{y.shape} weight={w.shape}')
    tin=x.shape[1];tout=y.shape[1];hdr=struct.pack('<8s10I16x',MAGIC,1,ci,co,k,tin,tout,int(pads[0]),int(dil[0]),int(stride[0]),group)
    with out.open('wb') as f:f.write(hdr);f.write(b.tobytes());f.write(pack_output_major(w).tobytes());f.write(x.tobytes());f.write(y.tobytes())
    meta={'node':n.name,'input_tensor':n.input[0],'output_tensor':n.output[0],'weight_shape':list(w.shape),'input_shape':list(xin.shape),'output_shape':list(yref.shape),'pad':int(pads[0]),'dilation':int(dil[0]),'stride':int(stride[0]),'group':group,'macs':int(ci*x.shape[1]*co*k)}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');probe.unlink(missing_ok=True);print('M33 real ConvTranspose packed:',json.dumps(meta,ensure_ascii=False));print('bundle:',out)
if __name__=='__main__':main()
