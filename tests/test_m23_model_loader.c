#include "dsasm_model.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc,char**argv){
    if(argc!=4){fprintf(stderr,"usage: %s fs2.dsfs aux.dsa lynx.dsn\n",argv[0]);return 2;}
    DSAsmAcousticModel m;
    int rc=ds_acoustic_model_load(&m,argv[1],argv[2],argv[3]);
    if(rc){fprintf(stderr,"model load failed rc=%d\n",rc);return 2;}
    printf("M23 loader: V=%u Cfs=%u Lfs=%u Hfs=%u | aux=%u->%u L=%u | rf I=%u Q=%u C=%u H=%u L=%u glu=%u\n",
        m.fs2.encoder.vocab_size,m.fs2.encoder.hidden_size,m.fs2.encoder.num_layers,m.fs2.encoder.num_heads,
        m.aux.input_dim,m.aux.output_dim,m.aux.num_layers,
        m.rf.input_dim,m.rf.condition_dim,m.rf.channels,m.rf.hidden_dim,m.rf.num_layers,m.rf.glu_type);
    if(!ds_acoustic_model_valid(&m)){fprintf(stderr,"cross-bundle validation failed\n");ds_acoustic_model_unload(&m);return 2;}

    const size_t P=5,T=16,D=m.rf.input_dim;
    int32_t tok[5]={1,2,3,4,5};
    int32_t mel2ph[16]={1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,0};
    float f0[16];for(size_t i=0;i<T;i++)f0[i]=(i+1<T)?220.0f+2.0f*(float)i:0.0f;
    float *noise=(float*)malloc(T*D*sizeof(float));float*out=(float*)malloc(T*D*sizeof(float));
    size_t wn=ds_acoustic_model_workspace_floats(&m,P,T);float*ws=(float*)malloc(wn*sizeof(float));
    if(!noise||!out||!ws||!wn){fprintf(stderr,"alloc/workspace failed\n");return 2;}
    for(size_t i=0;i<T*D;i++)noise[i]=(float)((int)(i%17)-8)*0.01f;
    float lo=-12.0f,hi=0.0f;
    DSAsmThreadPool*p=ds_threadpool_create(4);if(!p){fprintf(stderr,"pool failed\n");return 2;}
    rc=ds_acoustic_model_infer_f32_avx2(&m,tok,P,mel2ph,f0,T,noise,&lo,&hi,1,0.4f,1000.0f,2,out,ws,p);
    if(rc){fprintf(stderr,"infer failed rc=%d\n",rc);return 2;}
    float mx=0.0f;double sum=0.0;for(size_t i=0;i<T*D;i++){if(!isfinite(out[i])){fprintf(stderr,"nonfinite output\n");return 2;}float a=fabsf(out[i]);if(a>mx)mx=a;sum+=out[i];}
    float pad=0.0f;for(size_t d=0;d<D;d++){float a=fabsf(out[(T-1)*D+d]);if(a>pad)pad=a;}
    printf("M23 loader inference: workspace=%zu floats checksum=%.9g max_abs=%g pad=%g %s\n",wn,sum,mx,pad,pad==0.0f?"OK":"FAIL");
    ds_threadpool_destroy(p);free(noise);free(out);free(ws);ds_acoustic_model_unload(&m);
    return pad==0.0f?0:1;
}
