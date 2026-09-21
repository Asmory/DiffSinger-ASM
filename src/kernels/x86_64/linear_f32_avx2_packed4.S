.intel_syntax noprefix
.text

# void ds_linear_f32_avx2_packed4(
#     const float *x,          // rdi: [M, K]
#     const float *w_packed,   // rsi: packed 4-output blocks
#     const float *bias,       // rdx: [N]
#     float *y,                // rcx: [M, N]
#     size_t M,                // r8
#     size_t N,                // r9
#     size_t K                 // [rbp+16]
# );
#
# K must be divisible by 8. N may be arbitrary. The final output block is
# zero padded by the packer and only valid channels are stored.
#
# Packed layout, for every 4-output block and every K chunk of 8 floats:
#   W0[8], W1[8], W2[8], W3[8]
# = 32 floats / 128 bytes per K chunk.
#
# Computes Y = X @ W^T + bias.
# System V AMD64 ABI. AVX2 + FMA3 required.

.globl ds_linear_f32_avx2_packed4
.type ds_linear_f32_avx2_packed4, @function
ds_linear_f32_avx2_packed4:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r8                         # M
    mov r13, r9                         # N
    mov r14, QWORD PTR [rbp + 16]       # K

    test r12, r12
    jz .done
    test r13, r13
    jz .done
    test r14, 7
    jnz .done                           # contract K % 8 == 0

    # bytes per packed 4-output block = K/8 * 128 = K*16
    mov r15, r14
    shl r15, 4

.row_loop:
    xor r10d, r10d                      # n=0
    mov r11, rsi                        # packed block base

.block_loop:
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3

    mov rbx, r11
    xor eax, eax                        # k=0

.k_loop:
    cmp rax, r14
    jae .k_done

    vmovups ymm4, YMMWORD PTR [rdi + rax*4]
    vfmadd231ps ymm0, ymm4, YMMWORD PTR [rbx +  0]
    vfmadd231ps ymm1, ymm4, YMMWORD PTR [rbx + 32]
    vfmadd231ps ymm2, ymm4, YMMWORD PTR [rbx + 64]
    vfmadd231ps ymm3, ymm4, YMMWORD PTR [rbx + 96]

    add rbx, 128
    add rax, 8
    jmp .k_loop

.k_done:
    vextractf128 xmm4, ymm0, 1
    vaddps xmm0, xmm0, xmm4
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

    vextractf128 xmm4, ymm1, 1
    vaddps xmm1, xmm1, xmm4
    vhaddps xmm1, xmm1, xmm1
    vhaddps xmm1, xmm1, xmm1

    vextractf128 xmm4, ymm2, 1
    vaddps xmm2, xmm2, xmm4
    vhaddps xmm2, xmm2, xmm2
    vhaddps xmm2, xmm2, xmm2

    vextractf128 xmm4, ymm3, 1
    vaddps xmm3, xmm3, xmm4
    vhaddps xmm3, xmm3, xmm3
    vhaddps xmm3, xmm3, xmm3

    vaddss xmm0, xmm0, DWORD PTR [rdx + r10*4]
    vmovss DWORD PTR [rcx + r10*4], xmm0

    lea rax, [r10 + 1]
    cmp rax, r13
    jae .block_done
    vaddss xmm1, xmm1, DWORD PTR [rdx + rax*4]
    vmovss DWORD PTR [rcx + rax*4], xmm1

    lea rax, [r10 + 2]
    cmp rax, r13
    jae .block_done
    vaddss xmm2, xmm2, DWORD PTR [rdx + rax*4]
    vmovss DWORD PTR [rcx + rax*4], xmm2

    lea rax, [r10 + 3]
    cmp rax, r13
    jae .block_done
    vaddss xmm3, xmm3, DWORD PTR [rdx + rax*4]
    vmovss DWORD PTR [rcx + rax*4], xmm3

.block_done:
    add r10, 4
    add r11, r15
    cmp r10, r13
    jb .block_loop

    mov rax, r14
    shl rax, 2
    add rdi, rax                        # next X row

    mov rax, r13
    shl rax, 2
    add rcx, rax                        # next Y row

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
.size ds_linear_f32_avx2_packed4, .-ds_linear_f32_avx2_packed4

.section .note.GNU-stack,"",@progbits
