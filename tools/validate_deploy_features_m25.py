#!/usr/bin/env python3
from __future__ import annotations
import argparse,ctypes as C
from pathlib import Path
import numpy as np
import validate_pytorch_fs2_condition_m21 as A

PF=C.POINTER(C.c_float); PI=C.POINTER(C.c_int32)
LANG=1<<0;BREATH=1<<1;VOICE=1<<2;TENSION=1<<3;KEY=1<<4;SPEED=1<<5;SPK=1<<6;STRETCH=1<<7;LANGMASK=1<<8;FROZEN_SPK=1<<10;EXACT_ROPE=1<<11;RAW_DURATION=1<<12
class Extras(C.Structure):
    _fields_=[('num_languages',C.c_uint32),('flags',C.c_uint32),('language_embedding',PF),('language_token_mask',PF),
              ('breath_weight',PF),('breath_bias',PF),('voicing_weight',PF),('voicing_bias',PF),
              ('tension_weight',PF),('tension_bias',PF),('key_shift_weight',PF),('key_shift_bias',PF),
              ('speed_weight',PF),('speed_bias',PF),('stretch_table',PF),('frozen_speaker',PF),
              ('rope_cos',PF),('rope_sin',PF),('rope_max_tokens',C.c_uint32),
              ('breath_scale',C.c_float),('voicing_scale',C.c_float),
              ('tension_scale',C.c_float),('gender_clip_min',C.c_float),('gender_clip_max',C.c_float),
              ('gender_pre_scale',C.c_float),('key_shift_scale',C.c_float),('speed_clip_min',C.c_float),
              ('speed_clip_max',C.c_float),('speed_scale',C.c_float)]
class Inputs(C.Structure):
    _fields_=[('languages',PI),('breathiness',PF),('voicing',PF),('tension',PF),('gender',PF),('velocity',PF),('speaker_embedding_tc',PF)]

def pf(a):return a.ctypes.data_as(PF)
def pi(a):return a.ctypes.data_as(PI)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',type=Path,required=True);a=ap.parse_args()
    lib=A.setup(a.lib)
    lib.ds_adaptive_affine_f32_avx2.argtypes=[PF,PF,C.c_size_t,C.c_size_t]
    lib.ds_phoneme_mean_f32_avx2.argtypes=[PF,PI,PF,C.c_size_t,C.c_size_t]
    lib.ds_fs2_acoustic_condition_deploy_f32_avx2.argtypes=[C.POINTER(A.Acoustic),C.POINTER(Extras),C.POINTER(Inputs),PI,C.c_size_t,PI,PF,C.c_size_t,PF,PF,C.c_void_p]
    lib.ds_fs2_acoustic_condition_deploy_f32_avx2.restype=C.c_int
    seed=2525;P,T,Cc,L,H=17,48,128,2,2
    aw,keep,refs,emb,dw,db,fg,fb,tok,dur,m,f0,q=A.make_case(seed,P,T,Cc,L,H)
    base=A.ref_full(refs,emb,dw,db,fg,fb,tok,dur,m,f0,H,q)
    rng=np.random.default_rng(seed+500)
    primitive_x=np.ascontiguousarray(rng.standard_normal((5,128),dtype=np.float32))
    primitive_affine=np.ascontiguousarray(rng.standard_normal((5,256),dtype=np.float32))
    primitive_ref=primitive_x*primitive_affine[:,128:]+primitive_affine[:,:128]
    primitive_out=primitive_x.copy()
    lib.ds_adaptive_affine_f32_avx2(pf(primitive_out),pf(primitive_affine),5,128)
    primitive_err=float(np.max(np.abs(primitive_out-primitive_ref)))
    durations=np.array([2,0,3],np.int32);frames=np.ascontiguousarray(rng.standard_normal((5,128),dtype=np.float32));means=np.empty((3,128),np.float32)
    lib.ds_phoneme_mean_f32_avx2(pf(frames),pi(durations),pf(means),3,128)
    mean_ref=np.stack([frames[:2].mean(0),np.zeros(128,np.float32),frames[2:].mean(0)])
    mean_err=float(np.max(np.abs(means-mean_ref)))
    print(f'M25 adaptive primitives: affine_max_abs={primitive_err:.8g} mean_max_abs={mean_err:.8g} {"OK" if max(primitive_err,mean_err)<1e-5 else "FAIL"}')
    if max(primitive_err,mean_err)>=1e-5:raise SystemExit(2)
    nlang=5
    lang_emb=np.ascontiguousarray(rng.standard_normal((nlang,Cc)).astype(np.float32)*.025);lang_emb[0]=0
    langs=np.ascontiguousarray(rng.integers(1,nlang,size=P,dtype=np.int32));langs[tok==0]=0
    lang_mask=np.zeros(emb.shape[0],np.float32);lang_mask[1::2]=1.0
    # Rebuild encoder reference with language added at token input by using E internals is cumbersome;
    # construct a second reference case by directly reproducing encoder with modified embedding stage.
    import torch, torch.nn.functional as F, math
    E=A.E
    def enc_lang():
        x=np.zeros((P,Cc),np.float32);scale=np.sqrt(Cc).astype(np.float32) if hasattr(np.sqrt(Cc),'astype') else float(np.sqrt(Cc))
        for t in range(P):
            if tok[t]!=0:
                lid=langs[t] if lang_mask[tok[t]]>0.5 else 0
                x[t]=float(np.sqrt(Cc))*emb[tok[t]] + np.log1p(float(dur[t]))*dw + db + lang_emb[lid]
        # copy E.ref_forward layer loop logic but start from x
        xt=torch.from_numpy(x)
        mask=torch.from_numpy(tok==0)
        # refs produced by E.make_case are layer dictionaries used by E.ref_forward; easiest call local reconstruction below.
        hd=Cc//H
        def rope(z):
            half=hd//2; inv=torch.exp(-math.log(10000.0)*torch.arange(half,dtype=torch.float32)*2/hd)
            ang=torch.arange(P,dtype=torch.float32)[:,None]*inv[None];co=ang.cos();si=ang.sin()
            z=z.view(P,H,hd)
            a1=z[...,:half].clone();a2=z[...,half:].clone()
            z[...,:half]=a1*co[:,None,:]-a2*si[:,None,:]
            z[...,half:]=a2*co[:,None,:]+a1*si[:,None,:]
            return z.reshape(P,Cc)
        for r in refs:
            n=F.layer_norm(xt,(Cc,),torch.from_numpy(r['g1']),torch.from_numpy(r['b1']),1e-5)
            qkv=F.linear(n,torch.from_numpy(r['qkv']),torch.zeros(3*Cc));qq,kk,vv=qkv.split(Cc,-1);qq=rope(qq);kk=rope(kk)
            qh=qq.view(P,H,hd).transpose(0,1);kh=kk.view(P,H,hd).transpose(0,1);vh=vv.view(P,H,hd).transpose(0,1)
            sc=qh@kh.transpose(-1,-2)/math.sqrt(hd);sc=sc.masked_fill(mask[None,None,:],float('-inf'));att=torch.softmax(sc,-1);ao=(att@vh).transpose(0,1).reshape(P,Cc);ao[mask]=0
            xt=xt+F.linear(ao,torch.from_numpy(r['ow']),torch.zeros(Cc));xt[mask]=0
            n=F.layer_norm(xt,(Cc,),torch.from_numpy(r['g2']),torch.from_numpy(r['b2']),1e-5)
            z=F.conv1d(n.T[None],torch.from_numpy(r['fw1']),torch.from_numpy(r['fb1']),padding=1)[0].T*(3**-.5);z=F.gelu(z,approximate='none');z=F.linear(z,torch.from_numpy(r['fw2']),torch.from_numpy(r['fb2']));xt=xt+z;xt[mask]=0
        xt=F.layer_norm(xt,(Cc,),torch.from_numpy(fg),torch.from_numpy(fb),1e-5);xt[mask]=0
        return xt.detach().numpy()
    enc=enc_lang(); padded=np.concatenate([np.zeros((1,Cc),np.float32),enc],0);cond=padded[m].copy()
    st=A.stretch_ref(m,dur);se=A.sinemb(torch.round(1000*st),Cc);z=F.gelu(F.linear(se,torch.from_numpy(q['sw1']),torch.from_numpy(q['sb1'])),approximate='none');z=F.linear(z,torch.from_numpy(q['sw2']),torch.from_numpy(q['sb2']));cond+=z.numpy()
    gru=torch.nn.GRU(Cc,Cc,1,batch_first=True)
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.from_numpy(q['gwih']));gru.bias_ih_l0.copy_(torch.from_numpy(q['gbih']));gru.weight_hh_l0.copy_(torch.from_numpy(q['gwhh']));gru.bias_hh_l0.copy_(torch.from_numpy(q['gbhh']))
    gout,_=gru(torch.from_numpy(cond)[None]);cond+=gout[0].detach().numpy();pv=np.log1p(f0/700.0).astype(np.float32);cond+=pv[:,None]*q['pw'][None]+q['pb'][None]

    def rv(scale=.03):return np.ascontiguousarray(rng.standard_normal(Cc).astype(np.float32)*scale)
    bw,bb,vw,vb,tw,tb,kw,kb,sw,sb=[rv() for _ in range(10)]
    breath=np.ascontiguousarray(rng.uniform(-20,20,T).astype(np.float32));voice=np.ascontiguousarray(rng.uniform(-15,15,T).astype(np.float32));tens=np.ascontiguousarray(rng.uniform(-2,2,T).astype(np.float32));gender=np.ascontiguousarray(rng.uniform(-1.4,1.4,T).astype(np.float32));vel=np.ascontiguousarray(rng.uniform(.2,2.5,T).astype(np.float32));sp=np.ascontiguousarray(rng.standard_normal((T,Cc)).astype(np.float32)*.015)
    bs,vs,ts=1/96,1/96,.1;gmin,gmax,gpre,ks=-1.,1.,12.,1/12;smin,smax,ss=.1,5.,1.
    cond += (breath*bs)[:,None]*bw+bb;cond += (voice*vs)[:,None]*vw+vb;cond += (tens*ts)[:,None]*tw+tb
    gv=np.clip(gender,gmin,gmax)*gpre*ks;cond += gv[:,None]*kw+kb
    sv=np.clip(vel,smin,smax)*ss;cond += sv[:,None]*sw+sb;cond+=sp
    arrays=[lang_emb,lang_mask,langs,bw,bb,vw,vb,tw,tb,kw,kb,sw,sb,breath,voice,tens,gender,vel,sp];keep+=arrays
    ex=Extras(nlang,LANG|BREATH|VOICE|TENSION|KEY|SPEED|SPK|LANGMASK,pf(lang_emb),pf(lang_mask),pf(bw),pf(bb),pf(vw),pf(vb),pf(tw),pf(tb),pf(kw),pf(kb),pf(sw),pf(sb),PF(),PF(),PF(),PF(),0,bs,vs,ts,gmin,gmax,gpre,ks,smin,smax,ss)
    inp=Inputs(pi(langs),pf(breath),pf(voice),pf(tens),pf(gender),pf(vel),pf(sp))
    out=np.empty((T,Cc),np.float32);wn=lib.ds_fs2_acoustic_condition_workspace_floats(C.byref(aw),P,T);ws=np.empty(wn,np.float32);pool=lib.ds_threadpool_create(0)
    try:
        rc=lib.ds_fs2_acoustic_condition_deploy_f32_avx2(C.byref(aw),C.byref(ex),C.byref(inp),pi(tok),P,pi(m),pf(f0),T,pf(out),pf(ws),pool)
        if rc:raise RuntimeError(rc)
        e=np.abs(out-cond);ma=float(e.max());mr=float((e/np.maximum(np.abs(cond),1e-6)).max());print(f'M25 deploy extras: max_abs={ma:.8g} max_rel={mr:.8g} flags=0x{ex.flags:x} {"OK" if ma<2e-4 else "FAIL"}')
        if ma>=2e-4:raise SystemExit(2)
        # Deployment ONNX may constant-fold stretch MLP into a 1001xC lookup.
        ii=torch.arange(1001,dtype=torch.float32)
        ste=A.sinemb(ii,Cc)
        tab=F.linear(F.gelu(F.linear(ste,torch.from_numpy(q['sw1']),torch.from_numpy(q['sb1'])),approximate='none'),torch.from_numpy(q['sw2']),torch.from_numpy(q['sb2'])).numpy().astype(np.float32)
        keep.append(tab);ex.stretch_table=pf(tab);ex.flags|=STRETCH
        out2=np.empty_like(out)
        rc=lib.ds_fs2_acoustic_condition_deploy_f32_avx2(C.byref(aw),C.byref(ex),C.byref(inp),pi(tok),P,pi(m),pf(f0),T,pf(out2),pf(ws),pool)
        if rc:raise RuntimeError(rc)
        e2=np.abs(out2-cond);ma2=float(e2.max());print(f'M25 folded stretch table: max_abs={ma2:.8g} flags=0x{ex.flags:x} {"OK" if ma2<2e-4 else "FAIL"}')
        if ma2>=2e-4:raise SystemExit(2)

        # Frozen and external speaker embeddings are distinct graph contracts.
        # Keep a nonzero external input attached and prove the frozen-only flag
        # ignores it while broadcasting the model-owned vector over all frames.
        frozen=rv(.02);keep.append(frozen)
        ex.flags=(ex.flags&~SPK)|FROZEN_SPK;ex.frozen_speaker=pf(frozen)
        frozen_ref=cond-sp+frozen[None,:]
        out3=np.empty_like(out)
        rc=lib.ds_fs2_acoustic_condition_deploy_f32_avx2(C.byref(aw),C.byref(ex),C.byref(inp),pi(tok),P,pi(m),pf(f0),T,pf(out3),pf(ws),pool)
        if rc:raise RuntimeError(rc)
        e3=np.abs(out3-frozen_ref);ma3=float(e3.max());print(f'M25 frozen speaker: max_abs={ma3:.8g} flags=0x{ex.flags:x} {"OK" if ma3<2e-4 else "FAIL"}')
        if ma3>=2e-4:raise SystemExit(2)

        half=(Cc//H)//2
        inv=np.exp(-np.log(np.float32(10000.0))*np.arange(half,dtype=np.float32)*np.float32(2/(Cc//H)))
        angle=np.arange(P,dtype=np.float32)[:,None]*inv[None,:]
        rope_cos=np.ascontiguousarray(np.cos(angle),np.float32);rope_sin=np.ascontiguousarray(np.sin(angle),np.float32)
        keep.extend([rope_cos,rope_sin]);ex.rope_cos=pf(rope_cos);ex.rope_sin=pf(rope_sin);ex.rope_max_tokens=P;ex.flags|=EXACT_ROPE
        out4=np.empty_like(out)
        rc=lib.ds_fs2_acoustic_condition_deploy_f32_avx2(C.byref(aw),C.byref(ex),C.byref(inp),pi(tok),P,pi(m),pf(f0),T,pf(out4),pf(ws),pool)
        if rc:raise RuntimeError(rc)
        e4=np.abs(out4-frozen_ref);ma4=float(e4.max());print(f'M25 exact RoPE table: max_abs={ma4:.8g} flags=0x{ex.flags:x} {"OK" if ma4<2e-4 else "FAIL"}')
        if ma4>=2e-4:raise SystemExit(2)
    finally:lib.ds_threadpool_destroy(pool)
if __name__=='__main__':main()
