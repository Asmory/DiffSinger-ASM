#define _POSIX_C_SOURCE 200112L
#include "dsasm_threadpool.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void){
    const size_t n=1000003u;
    float *a=NULL,*b=NULL,*y=NULL;
    if(posix_memalign((void**)&a,64,n*sizeof(float)) ||
       posix_memalign((void**)&b,64,n*sizeof(float)) ||
       posix_memalign((void**)&y,64,n*sizeof(float))) return 2;
    for(size_t i=0;i<n;i++){
        a[i]=(float)((int)(i%113u)-56)*0.03125f;
        b[i]=(float)((int)(i%71u)-35)*0.015625f;
    }
    DSAsmThreadPool *p=ds_threadpool_create(8);
    if(!p) return 3;
    ds_threadpool_add_f32(p,a,b,y,n);
    float max_abs=0.0f;
    for(size_t i=0;i<n;i++){
        float d=fabsf(y[i]-(a[i]+b[i]));
        if(d>max_abs)max_abs=d;
    }
    printf("M43 parallel Add parity: n=%zu workers=%zu max_abs=%.9g\n",
           n,ds_threadpool_threads(p),max_abs);
    ds_threadpool_destroy(p);free(a);free(b);free(y);
    return max_abs==0.0f?0:1;
}
