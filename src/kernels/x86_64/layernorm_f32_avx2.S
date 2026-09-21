.intel_syntax noprefix
.text

# void ds_layernorm_f32_avx2(
#     const float *x,      // rdi [M,K]
#     const float *gamma,  // rsi [K]
#     const float *beta,   // rdx [K]
#     float *y,            // rcx [M,K]
#     size_t M,            // r8
#     size_t K,            // r9
#     float eps            // xmm0
# );
#
# Row-wise LayerNorm over K. Uses fp32 sum/sumsq accumulation, matching the
# inference-first nature of this project. Arbitrary K is supported.
# AVX2 + FMA3 required.

.globl ds_layernorm_f32_avx2
.type ds_layernorm_f32_avx2, @function
ds_layernorm_f32_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r8                         # rows remaining
    mov r13, r9                         # K
    vmovss xmm15, xmm0, xmm0            # eps
    test r12, r12
    jz .done
    test r13, r13
    jz .done

    mov r14, r13
    and r14, -8

.row_loop:
    vxorps ymm0, ymm0, ymm0             # sum
    vxorps ymm1, ymm1, ymm1             # sumsq
    xor eax, eax

.sum_vec:
    cmp rax, r14
    jae .sum_vec_done
    vmovups ymm2, YMMWORD PTR [rdi + rax*4]
    vaddps ymm0, ymm0, ymm2
    vfmadd231ps ymm1, ymm2, ymm2
    add rax, 8
    jmp .sum_vec

.sum_vec_done:
    # reduce vector sum -> scalar xmm0
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    # reduce vector sumsq -> scalar xmm1
    vextractf128 xmm2, ymm1, 1
    vaddps xmm1, xmm1, xmm2
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1

.sum_tail:
    cmp rax, r13
    jae .stats
    vmovss xmm2, DWORD PTR [rdi + rax*4]
    vaddss xmm0, xmm0, xmm2
    vfmadd231ss xmm1, xmm2, xmm2
    inc rax
    jmp .sum_tail

.stats:
    # invK
    vcvtsi2ss xmm2, xmm2, r13
    vmovss xmm3, DWORD PTR [rip + .L_one]
    vdivss xmm3, xmm3, xmm2            # invK

    vmulss xmm4, xmm0, xmm3            # mean
    vmulss xmm5, xmm1, xmm3            # E[x^2]
    vfnmadd231ss xmm5, xmm4, xmm4       # var = E[x^2] - mean^2
    vxorps xmm6, xmm6, xmm6
    vmaxss xmm5, xmm5, xmm6             # clamp roundoff-negative variance
    vaddss xmm5, xmm5, xmm15            # + eps
    vsqrtss xmm5, xmm5, xmm5
    vdivss xmm6, xmm3, xmm5             # NOTE xmm3=1/K, wrong if used directly
    # build inv_std = 1/sqrt(var+eps)
    vmovss xmm6, DWORD PTR [rip + .L_one]
    vdivss xmm6, xmm6, xmm5

    vbroadcastss ymm7, xmm4             # mean
    vbroadcastss ymm8, xmm6             # inv_std

    xor eax, eax
.out_vec:
    cmp rax, r14
    jae .out_vec_done
    vmovups ymm9, YMMWORD PTR [rdi + rax*4]
    vsubps ymm9, ymm9, ymm7
    vmulps ymm9, ymm9, ymm8
    vmovups ymm10, YMMWORD PTR [rsi + rax*4]
    vmovups ymm11, YMMWORD PTR [rdx + rax*4]
    vfmadd213ps ymm9, ymm10, ymm11      # y = norm*gamma + beta
    vmovups YMMWORD PTR [rcx + rax*4], ymm9
    add rax, 8
    jmp .out_vec

.out_vec_done:
    cmp rax, r13
    jae .next_row
.out_tail:
    vmovss xmm9, DWORD PTR [rdi + rax*4]
    vsubss xmm9, xmm9, xmm4
    vmulss xmm9, xmm9, xmm6
    vmulss xmm9, xmm9, DWORD PTR [rsi + rax*4]
    vaddss xmm9, xmm9, DWORD PTR [rdx + rax*4]
    vmovss DWORD PTR [rcx + rax*4], xmm9
    inc rax
    cmp rax, r13
    jb .out_tail

.next_row:
    mov rax, r13
    shl rax, 2
    add rdi, rax
    add rcx, rax
    dec r12
    jnz .row_loop

.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_layernorm_f32_avx2, .-ds_layernorm_f32_avx2

.section .rodata
.align 4
.L_one:
    .float 1.0
.section .note.GNU-stack,"",@progbits
