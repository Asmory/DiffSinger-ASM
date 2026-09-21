.intel_syntax noprefix
.text

# void ds_linear_residual_f32_avx2_m4n8(
#   const float *x,          // rdi [M,K]
#   const float *w_packed,   // rsi packed8: per N=8 block, W[8] for each k
#   const float *bias,       // rdx [N]
#   const float *residual,   // rcx [M,N]
#   float *y,                // r8  [M,N]
#   size_t M,                // r9
#   size_t N,                // [rbp+16]
#   size_t K                 // [rbp+24]
# );
# N must be divisible by 8. K arbitrary. Computes Y=XW^T+bias+residual.
# 4 rows x 8 output channels; each weight vector is reused across four rows.

.globl ds_linear_residual_f32_avx2_m4n8
.type ds_linear_residual_f32_avx2_m4n8, @function
ds_linear_residual_f32_avx2_m4n8:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    mov rax, QWORD PTR [rbp + 16]     # N
    mov r13, QWORD PTR [rbp + 24]     # K
    test r9, r9
    jz .done
    test rax, rax
    jz .done
    test r13, r13
    jz .done
    test rax, 7
    jnz .done

    mov r12, rax
    shr r12, 3                        # N blocks
    mov r14, r13
    shl r14, 2                        # x stride bytes
    mov r15, rax
    shl r15, 2                        # y/residual stride bytes
    mov QWORD PTR [rsp + 8], rsi

.main4_check:
    cmp r9, 4
    jb .tail_check
    mov QWORD PTR [rsp], r12

.nblock4:
    vmovups ymm0, YMMWORD PTR [rdx]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vmovaps ymm3, ymm0

    mov r10, rdi
    lea r11, [rdi + r14]
    lea rax, [r11 + r14]
    lea rbx, [rax + r14]
    mov r13, QWORD PTR [rbp + 24]

.k4:
    vmovups ymm8, YMMWORD PTR [rsi]

    vbroadcastss ymm9, DWORD PTR [r10]
    vfmadd231ps ymm0, ymm9, ymm8
    vbroadcastss ymm9, DWORD PTR [r11]
    vfmadd231ps ymm1, ymm9, ymm8
    vbroadcastss ymm9, DWORD PTR [rax]
    vfmadd231ps ymm2, ymm9, ymm8
    vbroadcastss ymm9, DWORD PTR [rbx]
    vfmadd231ps ymm3, ymm9, ymm8

    add r10, 4
    add r11, 4
    add rax, 4
    add rbx, 4
    add rsi, 32
    dec r13
    jnz .k4

    vaddps ymm0, ymm0, YMMWORD PTR [rcx]
    vmovups YMMWORD PTR [r8], ymm0
    vaddps ymm1, ymm1, YMMWORD PTR [rcx + r15]
    vmovups YMMWORD PTR [r8 + r15], ymm1
    lea r10, [r15 + r15]
    vaddps ymm2, ymm2, YMMWORD PTR [rcx + r10]
    vmovups YMMWORD PTR [r8 + r10], ymm2
    lea r10, [r15 + r15*2]
    vaddps ymm3, ymm3, YMMWORD PTR [rcx + r10]
    vmovups YMMWORD PTR [r8 + r10], ymm3

    add rdx, 32
    add rcx, 32
    add r8, 32
    dec QWORD PTR [rsp]
    jnz .nblock4

    mov rsi, QWORD PTR [rsp + 8]
    sub rdx, r15
    lea r10, [r15 + r15*2]
    add rcx, r10                       # old+N + 3N = next group row0
    add r8, r10
    lea r10, [r14 + r14*2]
    add rdi, r10
    add rdi, r14
    sub r9, 4
    jmp .main4_check

.tail_check:
    test r9, r9
    jz .done

.tail_row:
    mov QWORD PTR [rsp], r12

.nblock1:
    vmovups ymm0, YMMWORD PTR [rdx]
    mov r10, rdi
    mov r13, QWORD PTR [rbp + 24]
.k1:
    vmovups ymm8, YMMWORD PTR [rsi]
    vbroadcastss ymm9, DWORD PTR [r10]
    vfmadd231ps ymm0, ymm9, ymm8
    add r10, 4
    add rsi, 32
    dec r13
    jnz .k1
    vaddps ymm0, ymm0, YMMWORD PTR [rcx]
    vmovups YMMWORD PTR [r8], ymm0
    add rdx, 32
    add rcx, 32
    add r8, 32
    dec QWORD PTR [rsp]
    jnz .nblock1

    mov rsi, QWORD PTR [rsp + 8]
    sub rdx, r15
    add rdi, r14
    # rcx and r8 already moved one complete row
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
.size ds_linear_residual_f32_avx2_m4n8, .-ds_linear_residual_f32_avx2_m4n8

.section .note.GNU-stack,"",@progbits
