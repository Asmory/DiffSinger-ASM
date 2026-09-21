.intel_syntax noprefix
.text
# void ds_leaky_relu_f32_avx2(const float *x, float *y, size_t n, float alpha)
.globl ds_leaky_relu_f32_avx2
.type ds_leaky_relu_f32_avx2,@function
ds_leaky_relu_f32_avx2:
    test rdx, rdx
    jz .done
    vbroadcastss ymm5, xmm0
    vxorps ymm4, ymm4, ymm4
    mov rcx, rdx
    and rcx, -8
    xor r8d, r8d
.vec:
    cmp r8, rcx
    jae .tail
    vmovups ymm0, YMMWORD PTR [rdi+r8*4]
    vmaxps ymm1, ymm0, ymm4
    vminps ymm2, ymm0, ymm4
    vmulps ymm2, ymm2, ymm5
    vaddps ymm1, ymm1, ymm2
    vmovups YMMWORD PTR [rsi+r8*4], ymm1
    add r8, 8
    jmp .vec
.tail:
    cmp r8, rdx
    jae .done
    vmovss xmm0, DWORD PTR [rdi+r8*4]
    vxorps xmm4, xmm4, xmm4
    vmaxss xmm1, xmm0, xmm4
    vminss xmm2, xmm0, xmm4
    vmulss xmm2, xmm2, xmm5
    vaddss xmm1, xmm1, xmm2
    vmovss DWORD PTR [rsi+r8*4], xmm1
    inc r8
    jmp .tail
.done:
    vzeroupper
    ret
.size ds_leaky_relu_f32_avx2,.-ds_leaky_relu_f32_avx2
.section .note.GNU-stack,"",@progbits
