.intel_syntax noprefix
.text

# void ds_fused_linear_softsign_glu_f32_avx2(
#     const float *x,          // rdi: [M, K]
#     const float *w_left,     // rsi: [N, K]
#     const float *w_gate,     // rdx: [N, K]
#     const float *b_left,     // rcx: [N]
#     const float *b_gate,     // r8 : [N]
#     float *y,                // r9 : [M, N]
#     size_t M,                // [rbp+16]
#     size_t N,                // [rbp+24]
#     size_t K                 // [rbp+32]
# );
#
# Computes:
#   L = X @ W_left^T + b_left
#   G = X @ W_gate^T + b_gate
#   Y = L * (G / (1 + abs(G)))
#
# System V AMD64 ABI. AVX2 + FMA3 required.

.globl ds_fused_linear_softsign_glu_f32_avx2
.type ds_fused_linear_softsign_glu_f32_avx2, @function
ds_fused_linear_softsign_glu_f32_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, QWORD PTR [rbp + 16]      # rows remaining (M)
    mov r13, QWORD PTR [rbp + 24]      # N
    mov r14, QWORD PTR [rbp + 32]      # K

    test r12, r12
    jz .done
    test r13, r13
    jz .done

.row_loop:
    xor r15d, r15d                    # n = 0

.col_loop:
    # weight byte offset = n * K * sizeof(float)
    mov rax, r15
    imul rax, r14
    shl rax, 2
    lea r10, [rsi + rax]              # current W_left row
    lea r11, [rdx + rax]              # current W_gate row

    vxorps ymm0, ymm0, ymm0           # left accumulator
    vxorps ymm1, ymm1, ymm1           # gate accumulator

    mov rbx, r14
    and rbx, -8                        # vectorized K end
    xor eax, eax                       # k = 0

.k_vec_loop:
    cmp rax, rbx
    jae .k_vec_done

    vmovups ymm2, YMMWORD PTR [rdi + rax*4]
    vmovups ymm3, YMMWORD PTR [r10 + rax*4]
    vmovups ymm4, YMMWORD PTR [r11 + rax*4]
    vfmadd231ps ymm0, ymm2, ymm3
    vfmadd231ps ymm1, ymm2, ymm4

    add rax, 8
    jmp .k_vec_loop

.k_vec_done:
    # Horizontal reduce ymm0 -> xmm0[0]
    vextractf128 xmm2, ymm0, 1
    vaddps xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    # Horizontal reduce ymm1 -> xmm1[0]
    vextractf128 xmm2, ymm1, 1
    vaddps xmm1, xmm1, xmm2
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1

.k_tail_loop:
    cmp rax, r14
    jae .k_done

    vmovss xmm2, DWORD PTR [rdi + rax*4]
    vmovss xmm3, DWORD PTR [r10 + rax*4]
    vmovss xmm4, DWORD PTR [r11 + rax*4]
    vfmadd231ss xmm0, xmm2, xmm3
    vfmadd231ss xmm1, xmm2, xmm4

    inc rax
    jmp .k_tail_loop

.k_done:
    vaddss xmm0, xmm0, DWORD PTR [rcx + r15*4]
    vaddss xmm1, xmm1, DWORD PTR [r8  + r15*4]

    # softsign(g) = g / (1 + abs(g))
    vandps xmm2, xmm1, XMMWORD PTR [rip + .L_abs_mask]
    vaddss xmm2, xmm2, DWORD PTR [rip + .L_one]
    vdivss xmm1, xmm1, xmm2

    # y = left * softsign(gate)
    vmulss xmm0, xmm0, xmm1
    vmovss DWORD PTR [r9 + r15*4], xmm0

    inc r15
    cmp r15, r13
    jb .col_loop

    # Advance X by K floats and Y by N floats.
    mov rax, r14
    shl rax, 2
    add rdi, rax

    mov rax, r13
    shl rax, 2
    add r9, rax

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
.size ds_fused_linear_softsign_glu_f32_avx2, .-ds_fused_linear_softsign_glu_f32_avx2

.section .rodata
.align 16
.L_abs_mask:
    .long 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff
.L_one:
    .float 1.0

.section .note.GNU-stack,"",@progbits
