#!/usr/bin/env python3
import struct, numpy as np
from pathlib import Path

HDR = struct.Struct('<8s7I8Q28x')
TREC = struct.Struct('<4I4Q4Q')
OREC = struct.Struct('<8I16q4f16x')
assert HDR.size == 128 and TREC.size == 80 and OREC.size == 192
MAGIC=b'DSVOC35\0'
WORK=0; CONST=1; F32=1; RAW=2
OPS={'Add':1,'Sub':2,'Mul':3,'Div':4,'Mod':5,'LeakyRelu':6,'Tanh':7,'Sin':8,'CumSum':9,
     'Reshape':10,'Squeeze':11,'Transpose':12,'Slice':13,'Pad':14,'Conv':15,'ConvTranspose':16,'Unsqueeze':17}

def align(x,a): return (x+a-1)//a*a

class Builder:
    def __init__(self, frames, mel_bins, samples):
        self.frames=int(frames);self.mel_bins=int(mel_bins);self.samples=int(samples)
        self.tensors=[];self.ops=[];self.const=bytearray();self.arena=0
    def add_work(self, shape, alloc_nelem=None):
        shape=tuple(int(x) for x in shape); ne=int(np.prod(shape,dtype=np.int64)) if shape else 1
        al=int(alloc_nelem or ne); off=align(self.arena,16); self.arena=off+al
        self.tensors.append(dict(kind=WORK,dtype=F32,shape=shape,nelem=ne,alloc=al,offset=off,bytes=al*4))
        return len(self.tensors)-1
    def grow_work(self, tid, alloc_nelem):
        t=self.tensors[tid]
        if alloc_nelem<=t['alloc']: return
        # Re-home this tensor at the tail; old hole is intentionally kept in M35.
        off=align(self.arena,16);self.arena=off+int(alloc_nelem);t['offset']=off;t['alloc']=int(alloc_nelem);t['bytes']=int(alloc_nelem)*4
    def add_const(self, arr):
        arr=np.asarray(arr,dtype=np.float32,order='C'); raw=arr.tobytes(); off=align(len(self.const),64)
        if off>len(self.const):self.const.extend(b'\0'*(off-len(self.const)))
        self.const.extend(raw);shape=arr.shape;ne=arr.size
        self.tensors.append(dict(kind=CONST,dtype=F32,shape=shape,nelem=ne,alloc=ne,offset=off,bytes=len(raw)))
        return len(self.tensors)-1
    def add_blob(self, raw):
        raw=bytes(raw); off=align(len(self.const),64)
        if off>len(self.const): self.const.extend(b'\0'*(off-len(self.const)))
        self.const.extend(raw)
        self.tensors.append(dict(kind=CONST,dtype=RAW,shape=(len(raw),),nelem=len(raw),alloc=len(raw),offset=off,bytes=len(raw)))
        return len(self.tensors)-1
    def add_op(self, typ, ins, out, p=(), f=(), flags=0, reserved=0):
        ii=list(ins)+[0xffffffff]*3;pp=list(p)+[0]*16;ff=list(f)+[0.0]*4
        self.ops.append((OPS[typ],ii[0],ii[1],ii[2],out,len(self.tensors[out]['shape']),flags,int(reserved),*pp[:16],*ff[:4]))

    def plan_arena_lifetimes(self, mel_id, f0_id, out_id):
        """Repack WORK tensors with a linear-scan lifetime allocator.
        Returns (naive_floats, planned_floats). Inputs live from op 0; graph
        output is kept through the end. Blocks are reused only when the old
        lifetime ends strictly before the new tensor is produced, so an op
        can never overwrite one of its own inputs."""
        naive=self.arena
        nops=len(self.ops)
        prod=[None]*len(self.tensors)
        last=[-1]*len(self.tensors)
        for oi,op in enumerate(self.ops):
            in0,in1,in2,out=op[1],op[2],op[3],op[4]
            prod[out]=oi
            extra=(op[8+8] if (op[6]&8) else (op[7] if (op[6]&2) else 0xffffffff))
            for tid in (in0,in1,in2,extra):
                if tid!=0xffffffff and tid < len(last): last[tid]=max(last[tid],oi)
        last[out_id]=max(last[out_id],nops)
        items=[]
        for tid,t in enumerate(self.tensors):
            if t['kind']!=WORK: continue
            if tid in (mel_id,f0_id): start=0
            elif prod[tid] is not None: start=prod[tid]
            else: start=0
            end=max(last[tid],start)
            items.append((start,-t['alloc'],tid,end,t['alloc']))
        items.sort()
        active=[]  # (end,off,size)
        free=[]    # (size,off)
        tail=0
        for start,_,tid,end,size in items:
            keep=[]
            for e,off,sz in active:
                if e < start: free.append((sz,off))
                else: keep.append((e,off,sz))
            active=keep
            best=None
            for i,(sz,off) in enumerate(free):
                if sz>=size and (best is None or sz<free[best][0]): best=i
            if best is not None:
                sz,off=free.pop(best)
            else:
                off=align(tail,16);sz=size;tail=off+size
            self.tensors[tid]['offset']=off
            active.append((end,off,sz))
        self.arena=tail
        return naive,self.arena

    def write(self, path, mel_id, f0_id, out_id):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        toff=128;ooff=toff+len(self.tensors)*TREC.size;coff=align(ooff+len(self.ops)*OREC.size,64)
        hdr=HDR.pack(MAGIC,1,len(self.tensors),len(self.ops),mel_id,f0_id,out_id,0,
                     self.arena,len(self.const),toff,ooff,coff,self.frames,self.mel_bins,self.samples)
        with path.open('wb') as fp:
            fp.write(hdr)
            for t in self.tensors:
                dims=list(t['shape'])+[1]*4
                fp.write(TREC.pack(t['kind'],t['dtype'],len(t['shape']),0,*dims[:4],t['nelem'],t['alloc'],t['offset'],t['bytes']))
            for op in self.ops: fp.write(OREC.pack(*op))
            cur=fp.tell()
            if cur<coff:fp.write(b'\0'*(coff-cur))
            fp.write(self.const)
        return path
