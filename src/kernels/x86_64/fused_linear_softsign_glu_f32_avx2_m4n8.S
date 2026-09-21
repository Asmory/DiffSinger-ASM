.intel_syntax noprefix
.text

# void ds_fused_linear_softsign_glu_f32_avx2_m4n8(
#     const float *x,          // rdi [M,K]
#     const float *w_packed,   // rsi packed8, K-major within each N=8 block
#     const float *b_left,     // rdx [N]
#     const float *b_gate,     // rcx [N]
#     float *y,                // r8  [M,N]
#     size_t M,                // r9
#     size_t N,                // [rbp+16]
#     size_t K                 // [rbp+24]
# );
#
# Fast path for DiffSinger LYNXNet2 dimensions. N must be divisible by 8.
# K may be arbitrary. Weight packing for each 8-output N block:
#   for k in 0..K:
#       WL[n0:n0+8, k]   (8 floats / 32 B)
#       WG[n0:n0+8, k]   (8 floats / 32 B)
#
# The 4x8 micro-kernel computes four input rows and eight output channels at
# once. Each pair of 256-bit weight vectors is loaded once and reused by all
# four rows. Unlike packed4/M5, outputs are vectorized across N, so there is no
# horizontal reduction at the end of every dot product.
# AVX2 + FMA3, System V AMD64 ABI.

.globl ds_fused_linear_softsign_glu_f32_avx2_m4n8
.type ds_fused_linear_softsign_glu_f32_avx2_m4n8, @function
ds_fused_linear_softsign_glu_f32_avx2_m4n8:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16                       # [rsp]=N-blocks remaining, [rsp+8]=weight base

    mov rax, QWORD PTR [rbp + 16]     # N
    mov r13, QWORD PTR [rbp + 24]     # K (temporary)
    test r9, r9
    jz .done
    test rax, rax
    jz .done
    test r13, r13
    jz .done
    test rax, 7
    jnz .done                         # contract: N % 8 == 0

    mov r12, rax
    shr r12, 3                        # number of N=8 blocks
    mov r14, r13
    shl r14, 2                        # X row stride bytes = K*4
    mov r15, rax
    shl r15, 2                        # Y/bias row width bytes = N*4
    mov QWORD PTR [rsp + 8], rsi      # immutable packed-weight base

.main4_check:
    cmp r9, 4
    jb .tail_check

    mov QWORD PTR [rsp], r12

.nblock4:
    # Bias vectors seed the eight accumulators.
    vmovups ymm0, YMMWORD PTR [rdx]   # L row0
    vmovaps ymm1, ymm0                # L row1
    vmovaps ymm2, ymm0                # L row2
    vmovaps ymm3, ymm0                # L row3
    vmovups ymm4, YMMWORD PTR [rcx]   # G row0
    vmovaps ymm5, ymm4                # G row1
    vmovaps ymm6, ymm4                # G row2
    vmovaps ymm7, ymm4                # G row3

    # Four row pointers. Packed weight pointer rsi advances naturally.
    mov r10, rdi
    lea r11, [rdi + r14]
    lea rax, [r11 + r14]
    lea rbx, [rax + r14]
    mov r13, QWORD PTR [rbp + 24]     # k remaining

.k4:
    vmovups ymm8, YMMWORD PTR [rsi]
    vmovups ymm9, YMMWORD PTR [rsi + 32]

    vbroadcastss ymm10, DWORD PTR [r10]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm4, ymm10, ymm9

    vbroadcastss ymm10, DWORD PTR [r11]
    vfmadd231ps ymm1, ymm10, ymm8
    vfmadd231ps ymm5, ymm10, ymm9

    vbroadcastss ymm10, DWORD PTR [rax]
    vfmadd231ps ymm2, ymm10, ymm8
    vfmadd231ps ymm6, ymm10, ymm9

    vbroadcastss ymm10, DWORD PTR [rbx]
    vfmadd231ps ymm3, ymm10, ymm8
    vfmadd231ps ymm7, ymm10, ymm9

    add r10, 4
    add r11, 4
    add rax, 4
    add rbx, 4
    add rsi, 64
    dec r13
    jnz .k4

    # Vector softsign + gate multiply, row 0.
    vandps ymm10, ymm4, YMMWORD PTR [rip + .L_abs_mask8]
    vaddps ymm10, ymm10, YMMWORD PTR [rip + .L_one8]
    vdivps ymm4, ymm4, ymm10
    vmulps ymm0, ymm0, ymm4
    vmovups YMMWORD PTR [r8], ymm0

    # row 1
    vandps ymm10, ymm5, YMMWORD PTR [rip + .L_abs_mask8]
    vaddps ymm10, ymm10, YMMWORD PTR [rip + .L_one8]
    vdivps ymm5, ymm5, ymm10
    vmulps ymm1, ymm1, ymm5
    vmovups YMMWORD PTR [r8 + r15], ymm1

    # row 2
    vandps ymm10, ymm6, YMMWORD PTR [rip + .L_abs_mask8]
    vaddps ymm10, ymm10, YMMWORD PTR [rip + .L_one8]
    vdivps ymm6, ymm6, ymm10
    vmulps ymm2, ymm2, ymm6
    lea r10, [r15 + r15]
    vmovups YMMWORD PTR [r8 + r10], ymm2

    # row 3
    vandps ymm10, ymm7, YMMWORD PTR [rip + .L_abs_mask8]
    vaddps ymm10, ymm10, YMMWORD PTR [rip + .L_one8]
    vdivps ymm7, ymm7, ymm10
    vmulps ymm3, ymm3, ymm7
    lea r10, [r15 + r15*2]            # 3*y_stride
    vmovups YMMWORD PTR [r8 + r10], ymm3

    add rdx, 32
    add rcx, 32
    add r8, 32
    dec QWORD PTR [rsp]
    jnz .nblock4

    # Restore per-matrix pointers, then advance by four rows.
    mov rsi, QWORD PTR [rsp + 8]
    sub rdx, r15
    sub rcx, r15
    # r8 currently old_base + one row width; advance another 3 widths.
    lea r10, [r15 + r15*2]
    add r8, r10
    lea r10, [r14 + r14*2]
    add rdi, r10
    add rdi, r14                       # 4*x_stride
    sub r9, 4
    jmp .main4_check

.tail_check:
    test r9, r9
    jz .done

.tail_row:
    mov QWORD PTR [rsp], r12

.nblock1:
    vmovups ymm0, YMMWORD PTR [rdx]
    vmovups ymm4, YMMWORD PTR [rcx]
    mov r10, rdi
    mov r13, QWORD PTR [rbp + 24]

.k1:
    vmovups ymm8, YMMWORD PTR [rsi]
    vmovups ymm9, YMMWORD PTR [rsi + 32]
    vbroadcastss ymm10, DWORD PTR [r10]
    vfmadd231ps ymm0, ymm10, ymm8
    vfmadd231ps ymm4, ymm10, ymm9
    add r10, 4
    add rsi, 64
    dec r13
    jnz .k1

    vandps ymm10, ymm4, YMMWORD PTR [rip + .L_abs_mask8]
    vaddps ymm10, ymm10, YMMWORD PTR [rip + .L_one8]
    vdivps ymm4, ymm4, ymm10
    vmulps ymm0, ymm0, ymm4
    vmovups YMMWORD PTR [r8], ymm0

    add rdx, 32
    add rcx, 32
    add r8, 32
    dec QWORD PTR [rsp]
    jnz .nblock1

    mov rsi, QWORD PTR [rsp + 8]
    sub rdx, r15
    sub rcx, r15
    add rdi, r14
    # r8 already advanced exactly one output row.
    dec r9
    jnz .tail_row

.done:
    vzeroupper
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_fused_linear_softsign_glu_f32_avx2_m4n8, .-ds_fused_linear_softsign_glu_f32_avx2_m4n8

.section .rodata
.align 32
.L_abs_mask8:
    .long 0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff
    .long 0x7fffffff,0x7fffffff,0x7fffffff,0x7fffffff
.align 32
.L_one8:
    .float 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0

.section .note.GNU-stack,"",@progbits
