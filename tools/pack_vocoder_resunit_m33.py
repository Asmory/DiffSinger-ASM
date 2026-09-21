#!/usr/bin/env python3
import argparse,json,pathlib,struct
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
import onnxruntime as ort
MAGIC=b'DSVRU33\0'

def attrs(n):
    d={}
    for a in n.attribute:
        if a.type==onnx.AttributeProto.INT:d[a.name]=int(a.i)
        elif a.type==onnx.AttributeProto.INTS:d[a.name]=list(a.ints)
        elif a.type==onnx.AttributeProto.FLOAT:d[a.name]=float(a.f)
    return d

def pack4(w):
    co,ci,k=w.shape
    if co%4:raise RuntimeError('Cout must be multiple of 4')
    return w.reshape(co//4,4,ci,k).transpose(0,2,3,1).copy()

def exact(g,name):
    q=[n for n in g.node if n.name==name]
    if len(q)!=1:raise RuntimeError(f'{name}: matched {len(q)}')
    return q[0]

def add_output(g,name):
    if not any(o.name==name for o in g.output):g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,None))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('onnx');ap.add_argument('--prefix',default='/generator/resblocks.5');ap.add_argument('--unit',type=int,default=0);ap.add_argument('--frames',type=int,default=48);ap.add_argument('--f0',type=float,default=220.);ap.add_argument('--out',required=True);a=ap.parse_args()
    m=onnx.load(a.onnx);g=m.graph;init={x.name:numpy_helper.to_array(x) for x in g.initializer};prod={o:n for n in g.node for o in n.output};cons={}
    for n in g.node:
        for i in n.input:cons.setdefault(i,[]).append(n)
    n1=exact(g,f'{a.prefix}/convs1.{a.unit}/Conv');n2=exact(g,f'{a.prefix}/convs2.{a.unit}/Conv')
    p1=prod.get(n1.input[0]);p2=prod.get(n2.input[0])
    if p1 is None or p1.op_type!='LeakyRelu':raise RuntimeError(f'expected LeakyRelu before {n1.name}, got {None if p1 is None else p1.op_type}')
    if p2 is None or p2.op_type!='LeakyRelu' or n1.output[0] not in p2.input:raise RuntimeError('expected second LeakyRelu after conv1')
    xname=p1.input[0]
    adds=[n for n in cons.get(n2.output[0],[]) if n.op_type=='Add' and xname in n.input]
    if len(adds)!=1:raise RuntimeError(f'could not find residual Add for {n2.name}; matches={len(adds)}')
    add=adds[0];yname=add.output[0]
    def layer(n):
        if n.input[1] not in init:raise RuntimeError('weight not initializer')
        w=np.asarray(init[n.input[1]],np.float32);b=np.asarray(init[n.input[2]],np.float32) if len(n.input)>2 and n.input[2] in init else np.zeros(w.shape[0],np.float32);at=attrs(n);pads=at.get('pads',[0,0]);dil=at.get('dilations',[1]);strides=at.get('strides',[1]);group=at.get('group',1)
        if w.ndim!=3 or w.shape[0]!=w.shape[1] or w.shape[0]%4 or pads[0]!=pads[1] or strides!=[1] or group!=1:raise RuntimeError(f'unsupported residual Conv {n.name}: w={w.shape} pads={pads} stride={strides} group={group}')
        return w,b,int(pads[0]),int(dil[0])
    w1,b1,pad1,d1=layer(n1);w2,b2,pad2,d2=layer(n2);C=w1.shape[0]
    alpha1=attrs(p1).get('alpha',0.01);alpha2=attrs(p2).get('alpha',0.01)
    if abs(alpha1-alpha2)>1e-8:raise RuntimeError(f'LeakyReLU alpha mismatch {alpha1} {alpha2}')
    add_output(g,xname);add_output(g,yname);probe=pathlib.Path(a.out).with_suffix('.probe.onnx');onnx.save(m,probe)
    so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;s=ort.InferenceSession(str(probe),so,providers=['CPUExecutionProvider']);feeds={}
    for i in s.get_inputs():
        if i.name=='mel':feeds[i.name]=np.zeros((1,a.frames,128),np.float32)
        elif i.name=='f0':feeds[i.name]=np.full((1,a.frames),a.f0,np.float32)
        else:raise RuntimeError(f'unexpected input {i.name}')
    xin,yref=s.run([xname,yname],feeds);x=np.asarray(xin[0],np.float32);y=np.asarray(yref[0],np.float32)
    if x.shape!=y.shape or x.shape[0]!=C:raise RuntimeError(f'shape mismatch {x.shape}->{y.shape}, C={C}')
    T=x.shape[1];hdr=struct.pack('<8s9If16x',MAGIC,1,C,w1.shape[2],w2.shape[2],T,pad1,d1,pad2,d2,float(alpha1));out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:f.write(hdr);f.write(b1.tobytes());f.write(pack4(w1).tobytes());f.write(b2.tobytes());f.write(pack4(w2).tobytes());f.write(x.tobytes());f.write(y.tobytes())
    meta={'prefix':a.prefix,'unit':a.unit,'input_tensor':xname,'output_tensor':yname,'shape':list(xin.shape),'conv1':{'name':n1.name,'weight_shape':list(w1.shape),'pad':pad1,'dilation':d1},'conv2':{'name':n2.name,'weight_shape':list(w2.shape),'pad':pad2,'dilation':d2},'alpha':alpha1,'macs':int(C*T*C*(w1.shape[2]+w2.shape[2]))}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');probe.unlink(missing_ok=True);print('M33 real residual unit packed:',json.dumps(meta,ensure_ascii=False));print('bundle:',out)
if __name__=='__main__':main()
