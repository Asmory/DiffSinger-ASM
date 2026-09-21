.intel_syntax noprefix
.text

# float ds_dot_f32_avx2_fma(const float *a,const float *b,size_t n)
# rdi=a rsi=b rdx=n, xmm0 return. Arbitrary n.
.globl ds_dot_f32_avx2_fma
.type ds_dot_f32_avx2_fma,@function
ds_dot_f32_avx2_fma:
    vxorps ymm0,ymm0,ymm0
    xor rcx,rcx
    mov r8,rdx
    and r8,-8
.Lvec:
    cmp rcx,r8
    jae .Lreduce
    vmovups ymm1,YMMWORD PTR [rdi+rcx*4]
    vfmadd231ps ymm0,ymm1,YMMWORD PTR [rsi+rcx*4]
    add rcx,8
    jmp .Lvec
.Lreduce:
    vextractf128 xmm1,ymm0,1
    vaddps xmm0,xmm0,xmm1
    vhaddps xmm0,xmm0,xmm0
    vhaddps xmm0,xmm0,xmm0
.Ltail:
    cmp rcx,rdx
    jae .Ldone
    vmovss xmm1,DWORD PTR [rdi+rcx*4]
    vfmadd231ss xmm0,xmm1,DWORD PTR [rsi+rcx*4]
    inc rcx
    jmp .Ltail
.Ldone:
    vzeroupper
    ret
.size ds_dot_f32_avx2_fma,.-ds_dot_f32_avx2_fma
.section .note.GNU-stack,"",@progbits
