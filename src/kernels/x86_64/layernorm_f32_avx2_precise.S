.intel_syntax noprefix
.text

# void ds_layernorm_f32_avx2_precise(
#   const float *x, const float *gamma, const float *beta, float *y,
#   size_t M, size_t K, float eps);
#
# Accuracy-oriented two-pass LayerNorm for exported/PyTorch parity:
#   pass 1: mean
#   pass 2: sum((x-mean)^2)
#   pass 3: normalize/affine
# Avoids the cancellation-prone E[x^2]-mean^2 formula used by the fast kernel.
.globl ds_layernorm_f32_avx2_precise
.type ds_layernorm_f32_avx2_precise, @function
ds_layernorm_f32_avx2_precise:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r8
    mov r13, r9
    vmovss xmm15, xmm0, xmm0
    test r12, r12
    jz .done
    test r13, r13
    jz .done
    mov r14, r13
    and r14, -8

.row_loop:
    # pass 1: mean
    vxorps ymm0, ymm0, ymm0
    xor eax, eax
.sum_vec:
    cmp rax, r14
    jae .sum_vec_done
    vmovups ymm2, YMMWORD PTR [rdi + rax*4]
    vaddps ymm0, ymm0, ymm2
    add rax, 8
    jmp .sum_vec
.sum_vec_done:
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
.sum_tail:
    cmp rax, r13
    jae .mean
    vaddss xmm0, xmm0, DWORD PTR [rdi + rax*4]
    inc rax
    jmp .sum_tail
.mean:
    vcvtsi2ss xmm2, xmm2, r13
    vmovss xmm3, DWORD PTR [rip + .L_one]
    vdivss xmm3, xmm3, xmm2
    vmulss xmm4, xmm0, xmm3
    vbroadcastss ymm7, xmm4

    # pass 2: centered variance
    vxorps ymm1, ymm1, ymm1
    xor eax, eax
.var_vec:
    cmp rax, r14
    jae .var_vec_done
    vmovups ymm2, YMMWORD PTR [rdi + rax*4]
    vsubps ymm2, ymm2, ymm7
    vfmadd231ps ymm1, ymm2, ymm2
    add rax, 8
    jmp .var_vec
.var_vec_done:
    vextractf128 xmm2, ymm1, 1
    vaddps xmm1, xmm1, xmm2
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1
.var_tail:
    cmp rax, r13
    jae .invstd
    vmovss xmm2, DWORD PTR [rdi + rax*4]
    vsubss xmm2, xmm2, xmm4
    vfmadd231ss xmm1, xmm2, xmm2
    inc rax
    jmp .var_tail
.invstd:
    vmulss xmm5, xmm1, xmm3
    vaddss xmm5, xmm5, xmm15
    vsqrtss xmm5, xmm5, xmm5
    vmovss xmm6, DWORD PTR [rip + .L_one]
    vdivss xmm6, xmm6, xmm5
    vbroadcastss ymm8, xmm6

    # pass 3: affine
    xor eax, eax
.out_vec:
    cmp rax, r14
    jae .out_vec_done
    vmovups ymm9, YMMWORD PTR [rdi + rax*4]
    vsubps ymm9, ymm9, ymm7
    vmulps ymm9, ymm9, ymm8
    vmovups ymm10, YMMWORD PTR [rsi + rax*4]
    vmovups ymm11, YMMWORD PTR [rdx + rax*4]
    vfmadd213ps ymm9, ymm10, ymm11
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
.size ds_layernorm_f32_avx2_precise, .-ds_layernorm_f32_avx2_precise
.section .rodata
.align 4
.L_one:
    .float 1.0
.section .note.GNU-stack,"",@progbits
