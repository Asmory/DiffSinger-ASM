#!/usr/bin/env python3
import argparse,json,struct,pathlib,copy
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
import onnxruntime as ort
MAGIC=b'DSVN39\0\0'

def attrs(n):
 d={}
 for a in n.attribute:
  if a.type==onnx.AttributeProto.INT:d[a.name]=int(a.i)
  elif a.type==onnx.AttributeProto.INTS:d[a.name]=list(a.ints)
 return d

def select(g,name):
 q=[n for n in g.node if n.op_type=='Conv' and n.name==name]
 if len(q)!=1:raise RuntimeError(f'{name!r} matched {len(q)} Conv nodes')
 return q[0]

def addout(g,name):
 if not any(o.name==name for o in g.output):g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,None))

def pack8f(w):
 co,ci,k=w.shape
 assert co%8==0
 out=np.zeros((co//8,ci,k,8),np.float32)
 for oc in range(co):out[oc//8,:,:,oc%8]=w[oc]
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('onnx');ap.add_argument('--node',required=True);ap.add_argument('--out',required=True);ap.add_argument('--frames',type=int,default=48);a=ap.parse_args()
 out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 m=onnx.load(a.onnx);g=m.graph;init={x.name:numpy_helper.to_array(x) for x in g.initializer};n=select(g,a.node);ad=attrs(n)
 W=np.asarray(init[n.input[1]],np.float32);co,ci,k=map(int,W.shape);b=np.asarray(init[n.input[2]],np.float32) if len(n.input)>2 and n.input[2] in init else np.zeros(co,np.float32)
 if co%8:raise RuntimeError('Cout must be divisible by 8')
 stride=ad.get('strides',[1]);dil=ad.get('dilations',[1]);pads=ad.get('pads',[0,0]);group=ad.get('group',1)
 if stride!=[1] or group!=1 or len(dil)!=1 or len(pads)!=2 or pads[0]!=pads[1]:raise RuntimeError(f'unsupported attrs {ad}')
 addout(g,n.input[0]);addout(g,n.output[0]);probe=out.with_suffix('.probe.onnx');onnx.save(m,probe)
 so=ort.SessionOptions();so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL;s=ort.InferenceSession(str(probe),so,providers=['CPUExecutionProvider'])
 feed={}
 for inp in s.get_inputs():
  if inp.name=='mel':feed[inp.name]=np.zeros((1,a.frames,128),np.float32)
  elif inp.name=='f0':feed[inp.name]=np.full((1,a.frames),220.,np.float32)
  else:raise RuntimeError(f'unexpected input {inp.name}')
 xin,yref=s.run([n.input[0],n.output[0]],feed);x=np.asarray(xin[0],np.float32);y=np.asarray(yref[0],np.float32)
 tin=int(x.shape[1]);tout=int(y.shape[1]);pad=int(pads[0]);dd=int(dil[0]);kred=ci*k;k4=(kred+3)//4;tblocks=(tout+7)//8
 if tout%8:raise RuntimeError('M39 VNNI lab currently requires Tout % 8 == 0')
 # per-output-channel symmetric int8 weight quantization
 ws=np.empty(co,np.float32);qw=np.zeros_like(W,dtype=np.int8);corr=np.empty(co,np.int32)
 for oc in range(co):
  mx=float(np.max(np.abs(W[oc])));sc=mx/127. if mx>0 else 1.;ws[oc]=sc
  q=np.rint(W[oc]/sc).clip(-127,127).astype(np.int8);qw[oc]=q;corr[oc]=-128*int(q.astype(np.int32).sum())
 # [ob,k4,oc8,4]
 wp=np.zeros((co//8,k4,8,4),np.int8)
 for oc in range(co):
  flat=qw[oc].reshape(-1)
  for r,v in enumerate(flat):wp[oc//8,r//4,oc%8,r%4]=v
 fp=pack8f(W)
 hdr=struct.pack('<8s12I8x',MAGIC,1,ci,co,k,tin,tout,pad,dd,k4,tblocks,0,0)
 with open(out,'wb') as f:
  f.write(hdr);f.write(b.astype(np.float32).tobytes());f.write(ws.tobytes());f.write(corr.tobytes());f.write(wp.tobytes());f.write(fp.tobytes());f.write(x.tobytes());f.write(y.tobytes())
 meta={'node':n.name,'shape_in':list(xin.shape),'shape_out':list(yref.shape),'weight_shape':list(W.shape),'pad':pad,'dilation':dd,'kred':kred,'k4':k4,'tblocks':tblocks,'int8_weight_bytes':int(wp.nbytes),'fp32_weight_bytes':int(fp.nbytes)}
 out.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 try:probe.unlink()
 except OSError:pass
 print('M39 VNNI hot Conv packed:',json.dumps(meta))
 print('bundle:',out)
if __name__=='__main__':main()
