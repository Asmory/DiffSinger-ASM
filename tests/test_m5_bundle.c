#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ds_layernorm_f32_avx2(const float*, const float*, const float*, float*, size_t, size_t, float);
void ds_depthwise_conv1d_k31_tc_f32_avx2(const float*, const float*, const float*, float*, size_t, size_t);
void ds_fused_linear_softsign_glu_f32_avx2_packed4(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);
void ds_linear_residual_f32_avx2_packed4(const float*, const float*, const float*, const float*, float*, size_t, size_t, size_t);

static size_t align64(size_t x) { return (x + 63u) & ~(size_t)63u; }

static void *read_all(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    void *p = malloc((size_t)n ? (size_t)n : 1);
    if (!p) { fclose(f); return NULL; }
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return NULL; }
    fclose(f); *size_out = (size_t)n; return p;
}

static uint32_t u32le(const unsigned char *p) {
    uint32_t v; memcpy(&v, p, sizeof(v)); return v;
}

static const float *take(const unsigned char *blob, size_t blob_n, size_t *off, size_t floats, const char *name) {
    *off = align64(*off);
    size_t bytes = floats * sizeof(float);
    if (*off > blob_n || bytes > blob_n - *off) {
        fprintf(stderr, "bundle truncated at %s\n", name); exit(2);
    }
    const float *p = (const float *)(blob + *off);
    *off += bytes;
    return p;
}

static float max_abs(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i=0;i<n;i++) { float d=fabsf(a[i]-b[i]); if (d>m) m=d; }
    return m;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s block.dsb test_input.f32 test_output_pytorch.f32\n", argv[0]);
        return 2;
    }
    size_t bn=0,xn=0,yn=0;
    unsigned char *blob = read_all(argv[1], &bn);
    float *x = read_all(argv[2], &xn);
    float *ref = read_all(argv[3], &yn);
    if (!blob || !x || !ref) return 2;
    if (bn < 64 || memcmp(blob, "DSBLK5\0\0", 8) != 0) {
        fprintf(stderr, "invalid DSBLK5 bundle\n"); return 2;
    }
    uint32_t version=u32le(blob+8), D=u32le(blob+12), H=u32le(blob+16), K=u32le(blob+20);
    if (version != 1 || K != 31 || !D || !H || D%8 || H%8) {
        fprintf(stderr, "unsupported bundle v=%u D=%u H=%u K=%u\n",version,D,H,K); return 2;
    }
    if (xn != yn || xn % (D*sizeof(float)) != 0) {
        fprintf(stderr, "bad test vector size\n"); return 2;
    }
    size_t T = xn / (D*sizeof(float));

    size_t off=64;
    const float *gamma=take(blob,bn,&off,D,"gamma");
    const float *beta=take(blob,bn,&off,D,"beta");
    const float *dw=take(blob,bn,&off,31ull*D,"dw");
    const float *dw_b=take(blob,bn,&off,D,"dw_b");
    size_t hblocks=(H+3u)/4u, dblocks=(D+3u)/4u;
    const float *p1=take(blob,bn,&off,hblocks*(size_t)D*8u,"p1");
    const float *b1l=take(blob,bn,&off,H,"b1l");
    const float *b1g=take(blob,bn,&off,H,"b1g");
    const float *p2=take(blob,bn,&off,hblocks*(size_t)H*8u,"p2");
    const float *b2l=take(blob,bn,&off,H,"b2l");
    const float *b2g=take(blob,bn,&off,H,"b2g");
    const float *p3=take(blob,bn,&off,dblocks*(size_t)H*4u,"p3");
    const float *b3=take(blob,bn,&off,D,"b3");

    float *ln=malloc(T*(size_t)D*4), *dwo=malloc(T*(size_t)D*4);
    float *h1=malloc(T*(size_t)H*4), *h2=malloc(T*(size_t)H*4), *y=malloc(T*(size_t)D*4);
    if(!ln||!dwo||!h1||!h2||!y) return 2;

    ds_layernorm_f32_avx2(x,gamma,beta,ln,T,D,1e-5f);
    ds_depthwise_conv1d_k31_tc_f32_avx2(ln,dw,dw_b,dwo,T,D);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(dwo,p1,b1l,b1g,h1,T,H,D);
    ds_fused_linear_softsign_glu_f32_avx2_packed4(h1,p2,b2l,b2g,h2,T,H,H);
    ds_linear_residual_f32_avx2_packed4(h2,p3,b3,x,y,T,D,H);

    float err=max_abs(ref,y,T*(size_t)D);
    printf("DSBLK5 PyTorch-reference check: T=%zu D=%u H=%u max_abs=%g %s\n",T,D,H,err,err<1e-4f?"OK":"FAIL");

    free(blob);free(x);free(ref);free(ln);free(dwo);free(h1);free(h2);free(y);
    return err<1e-4f?0:1;
}
