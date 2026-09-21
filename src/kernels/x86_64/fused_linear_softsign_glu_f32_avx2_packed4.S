.intel_syntax noprefix
.text

# void ds_fused_linear_softsign_glu_f32_avx2_packed4(
#     const float *x,          // rdi: [M, K]
#     const float *w_packed,   // rsi: packed 4-output blocks
#     const float *b_left,     // rdx: [N]
#     const float *b_gate,     // rcx: [N]
#     float *y,                // r8 : [M, N]
#     size_t M,                // r9
#     size_t N,                // [rbp+16]
#     size_t K                 // [rbp+24]
# );
#
# K must be a multiple of 8. N may be arbitrary; the final packed block is
# zero padded to four output channels and only valid outputs are stored.
#
# Packed format, for every 4-output block and every 8-float K chunk:
#   L0[8], L1[8], L2[8], L3[8], G0[8], G1[8], G2[8], G3[8]
# i.e. 64 floats / 256 bytes per K chunk.
#
# This kernel reuses one X vector across eight FMAs (4 left + 4 gate),
# avoiding the M1 kernel's reload of X for every output channel.
# System V AMD64 ABI. AVX2 + FMA3 required.

.globl ds_fused_linear_softsign_glu_f32_avx2_packed4
.type ds_fused_linear_softsign_glu_f32_avx2_packed4, @function
ds_fused_linear_softsign_glu_f32_avx2_packed4:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r9                        # M rows remaining
    mov r13, QWORD PTR [rbp + 16]      # N
    mov r14, QWORD PTR [rbp + 24]      # K

    test r12, r12
    jz .done
    test r13, r13
    jz .done
    test r14, 7
    jnz .done                          # contract: K % 8 == 0

    # Bytes occupied by one packed 4-output block:
    # K floats * 8 planes * sizeof(float) = K * 32 bytes.
    mov r9, r14
    shl r9, 5

.row_loop:
    xor r10d, r10d                     # n = 0
    mov r11, rsi                       # packed block pointer

.block_loop:
    vxorps ymm0, ymm0, ymm0            # left 0
    vxorps ymm1, ymm1, ymm1            # left 1
    vxorps ymm2, ymm2, ymm2            # left 2
    vxorps ymm3, ymm3, ymm3            # left 3
    vxorps ymm4, ymm4, ymm4            # gate 0
    vxorps ymm5, ymm5, ymm5            # gate 1
    vxorps ymm6, ymm6, ymm6            # gate 2
    vxorps ymm7, ymm7, ymm7            # gate 3

    mov rbx, r11                       # current packed K chunk
    xor eax, eax                       # k = 0

.k_loop:
    cmp rax, r14
    jae .k_done

    vmovups ymm8, YMMWORD PTR [rdi + rax*4]

    vfmadd231ps ymm0, ymm8, YMMWORD PTR [rbx +   0]
    vfmadd231ps ymm1, ymm8, YMMWORD PTR [rbx +  32]
    vfmadd231ps ymm2, ymm8, YMMWORD PTR [rbx +  64]
    vfmadd231ps ymm3, ymm8, YMMWORD PTR [rbx +  96]
    vfmadd231ps ymm4, ymm8, YMMWORD PTR [rbx + 128]
    vfmadd231ps ymm5, ymm8, YMMWORD PTR [rbx + 160]
    vfmadd231ps ymm6, ymm8, YMMWORD PTR [rbx + 192]
    vfmadd231ps ymm7, ymm8, YMMWORD PTR [rbx + 224]

    add rbx, 256
    add rax, 8
    jmp .k_loop

.k_done:
    # Horizontal reductions: each ymm accumulator -> low scalar.
    vextractf128 xmm8, ymm0, 1
    vaddps xmm0, xmm0, xmm8
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    vextractf128 xmm8, ymm1, 1
    vaddps xmm1, xmm1, xmm8
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1

    vextractf128 xmm8, ymm2, 1
    vaddps xmm2, xmm2, xmm8
    vhaddps xmm2, xmm2, xmm2
    vhaddps xmm2, xmm2, xmm2

    vextractf128 xmm8, ymm3, 1
    vaddps xmm3, xmm3, xmm8
    vhaddps xmm3, xmm3, xmm3
    vhaddps xmm3, xmm3, xmm3

    vextractf128 xmm8, ymm4, 1
    vaddps xmm4, xmm4, xmm8
    vhaddps xmm4, xmm4, xmm4
    vhaddps xmm4, xmm4, xmm4

    vextractf128 xmm8, ymm5, 1
    vaddps xmm5, xmm5, xmm8
    vhaddps xmm5, xmm5, xmm5
    vhaddps xmm5, xmm5, xmm5

    vextractf128 xmm8, ymm6, 1
    vaddps xmm6, xmm6, xmm8
    vhaddps xmm6, xmm6, xmm6
    vhaddps xmm6, xmm6, xmm6

    vextractf128 xmm8, ymm7, 1
    vaddps xmm7, xmm7, xmm8
    vhaddps xmm7, xmm7, xmm7
    vhaddps xmm7, xmm7, xmm7

    # Bias. Padded output lanes have no valid bias, so only load when the
    # corresponding n+i < N. Invalid lanes are never stored.
    vaddss xmm0, xmm0, DWORD PTR [rdx + r10*4]
    vaddss xmm4, xmm4, DWORD PTR [rcx + r10*4]

    lea rax, [r10 + 1]
    cmp rax, r13
    jae .activate0
    vaddss xmm1, xmm1, DWORD PTR [rdx + rax*4]
    vaddss xmm5, xmm5, DWORD PTR [rcx + rax*4]

    lea rax, [r10 + 2]
    cmp rax, r13
    jae .activate0
    vaddss xmm2, xmm2, DWORD PTR [rdx + rax*4]
    vaddss xmm6, xmm6, DWORD PTR [rcx + rax*4]

    lea rax, [r10 + 3]
    cmp rax, r13
    jae .activate0
    vaddss xmm3, xmm3, DWORD PTR [rdx + rax*4]
    vaddss xmm7, xmm7, DWORD PTR [rcx + rax*4]

.activate0:
    # softsign(g) = g / (1 + abs(g)); y = left * softsign(g)
    vandps xmm8, xmm4, XMMWORD PTR [rip + .L_abs_mask]
    vaddss xmm8, xmm8, DWORD PTR [rip + .L_one]
    vdivss xmm4, xmm4, xmm8
    vmulss xmm0, xmm0, xmm4
    vmovss DWORD PTR [r8 + r10*4], xmm0

    lea rax, [r10 + 1]
    cmp rax, r13
    jae .block_done
    vandps xmm8, xmm5, XMMWORD PTR [rip + .L_abs_mask]
    vaddss xmm8, xmm8, DWORD PTR [rip + .L_one]
    vdivss xmm5, xmm5, xmm8
    vmulss xmm1, xmm1, xmm5
    vmovss DWORD PTR [r8 + rax*4], xmm1

    lea rax, [r10 + 2]
    cmp rax, r13
    jae .block_done
    vandps xmm8, xmm6, XMMWORD PTR [rip + .L_abs_mask]
    vaddss xmm8, xmm8, DWORD PTR [rip + .L_one]
    vdivss xmm6, xmm6, xmm8
    vmulss xmm2, xmm2, xmm6
    vmovss DWORD PTR [r8 + rax*4], xmm2

    lea rax, [r10 + 3]
    cmp rax, r13
    jae .block_done
    vandps xmm8, xmm7, XMMWORD PTR [rip + .L_abs_mask]
    vaddss xmm8, xmm8, DWORD PTR [rip + .L_one]
    vdivss xmm7, xmm7, xmm8
    vmulss xmm3, xmm3, xmm7
    vmovss DWORD PTR [r8 + rax*4], xmm3

.block_done:
    add r10, 4
    add r11, r9
    cmp r10, r13
    jb .block_loop

    # Next row.
    mov rax, r14
    shl rax, 2
    add rdi, rax

    mov rax, r13
    shl rax, 2
    add r8, rax

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
.size ds_fused_linear_softsign_glu_f32_avx2_packed4, .-ds_fused_linear_softsign_glu_f32_avx2_packed4

.section .rodata
.align 16
.L_abs_mask:
    .long 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
.L_one:
    .float 1.0

.section .note.GNU-stack,"",@progbits
