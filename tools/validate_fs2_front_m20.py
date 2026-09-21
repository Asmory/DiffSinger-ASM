#!/usr/bin/env python3
import argparse,ctypes,math
import numpy as np
import torch
FP=ctypes.POINTER(ctypes.c_float);IP=ctypes.POINTER(ctypes.c_int32)
def fp(a):return a.ctypes.data_as(FP)
def ip(a):return a.ctypes.data_as(IP)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',default='build/libdsasm_m20.so');a=ap.parse_args();lib=ctypes.CDLL(a.lib)
    lib.ds_fs2_mel2ph_to_dur_i32.argtypes=[IP,ctypes.c_size_t,ctypes.c_size_t,IP];lib.ds_fs2_mel2ph_to_dur_i32.restype=ctypes.c_int
    lib.ds_fs2_stretch_f32.argtypes=[IP,ctypes.c_size_t,IP,ctypes.c_size_t,FP];lib.ds_fs2_stretch_f32.restype=ctypes.c_int
    lib.ds_fs2_gather_encoder_f32.argtypes=[FP,IP,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,FP];lib.ds_fs2_gather_encoder_f32.restype=ctypes.c_int
    lib.ds_fs2_pitch_input_f32.argtypes=[FP,ctypes.c_size_t,FP];lib.ds_fs2_duration_input_f32.argtypes=[IP,ctypes.c_size_t,FP]
    mel=np.array([1,1,2,2,2,2,3,3,3,0,0],np.int32);P=3;T=len(mel)
    dur=np.zeros(P,np.int32);assert lib.ds_fs2_mel2ph_to_dur_i32(ip(mel),T,P,ip(dur))==0
    ref=np.array([2,4,3],np.int32);assert np.array_equal(dur,ref)
    st=np.zeros(T,np.float32);assert lib.ds_fs2_stretch_f32(ip(mel),T,ip(dur),P,fp(st))==0
    # exact upstream formula for B=1
    mt=torch.from_numpy(mel.astype(np.int64))[None];dt=torch.from_numpy(dur.astype(np.int64))[None]
    dcat=torch.cat([torch.ones_like(dt[:,:1]),dt],1);m2d=torch.gather(dcat,1,mt);bm=mt[:,1:]>mt[:,:-1];sd=1-bm*m2d[:,:-1];sd=torch.nn.functional.pad(sd,[1,0]);sr=torch.cumsum(sd,1).float()/m2d;sr=sr*(mt>0)
    se=float(np.max(np.abs(st-sr[0].numpy())));print(f'M20 FS2 stretch max_abs={se:g} dur={dur.tolist()} OK');assert se==0
    H=16;enc=np.arange(P*H,dtype=np.float32).reshape(P,H)/17;got=np.zeros((T,H),np.float32);assert lib.ds_fs2_gather_encoder_f32(fp(enc),ip(mel),T,P,H,fp(got))==0
    gr=np.zeros_like(got)
    for t,q in enumerate(mel):
        if q:gr[t]=enc[q-1]
    assert np.array_equal(got,gr);print('M20 FS2 mel2ph gather max_abs=0 OK')
    f0=np.array([0,65,220,440,880],np.float32);po=np.empty_like(f0);lib.ds_fs2_pitch_input_f32(fp(f0),len(f0),fp(po));pr=np.log1p(f0/700).astype(np.float32);pe=float(np.max(np.abs(po-pr)));print(f'M20 FS2 pitch-log max_abs={pe:g} OK');assert pe<2e-7
    do=np.empty(P,np.float32);lib.ds_fs2_duration_input_f32(ip(dur),P,fp(do));dr=np.log1p(dur.astype(np.float32));de=float(np.max(np.abs(do-dr)));print(f'M20 FS2 duration-log max_abs={de:g} OK');assert de<2e-7
if __name__=='__main__':main()
