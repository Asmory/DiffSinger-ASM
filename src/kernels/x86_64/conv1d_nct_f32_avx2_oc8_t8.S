.intel_syntax noprefix
.text

# M36 Conv1d: 8 output channels x 8 time samples.
# Packed weights: [Cout/8, Cin, K, 8]. Input is already time padded.
# void ds_conv1d_nct_f32_avx2_oc8_t8(
#   const float *x_padded, const float *w_packed8, const float *bias, float *y,
#   size_t Cin, size_t Cout, size_t K, size_t Tout, size_t Tp, size_t dilation);
.globl ds_conv1d_nct_f32_avx2_oc8_t8
.type ds_conv1d_nct_f32_avx2_oc8_t8,@function
ds_conv1d_nct_f32_avx2_oc8_t8:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40

    test r8, r8
    jz .done
    test r9, r9
    jz .done
    test r9, 7
    jnz .done

    mov r12, QWORD PTR [rbp+16]      # K
    mov r13, QWORD PTR [rbp+24]      # Tout
    mov r14, QWORD PTR [rbp+32]      # Tp
    shl r14, 2                       # Tp bytes
    mov r15, r8                      # Cin

    mov QWORD PTR [rsp+0], rdi       # original x
    mov rax, QWORD PTR [rbp+40]      # dilation
    shl rax, 2
    mov QWORD PTR [rsp+8], rax
    mov rax, r15
    imul rax, r12
    shl rax, 5                       # Cin*K*8 floats = *32 bytes
    mov QWORD PTR [rsp+16], rax
    mov rax, r13
    and rax, -8
    mov QWORD PTR [rsp+24], rax      # T8
    mov rbx, r9
    shr rbx, 3                       # output blocks

.ob_loop:
    xor r10d, r10d
.t8_check:
    cmp r10, QWORD PTR [rsp+24]
    jae .tail_check

    vbroadcastss ymm0, DWORD PTR [rdx+0]
    vbroadcastss ymm1, DWORD PTR [rdx+4]
    vbroadcastss ymm2, DWORD PTR [rdx+8]
    vbroadcastss ymm3, DWORD PTR [rdx+12]
    vbroadcastss ymm4, DWORD PTR [rdx+16]
    vbroadcastss ymm5, DWORD PTR [rdx+20]
    vbroadcastss ymm6, DWORD PTR [rdx+24]
    vbroadcastss ymm7, DWORD PTR [rdx+28]

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    mov r9, r15
.ci_vec:
    mov rdi, r11
    mov r8, r12
.k_vec:
    vmovups ymm8, YMMWORD PTR [rdi]
    vbroadcastss ymm9, DWORD PTR [rax+0]
    vfmadd231ps ymm0, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+4]
    vfmadd231ps ymm1, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+8]
    vfmadd231ps ymm2, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+12]
    vfmadd231ps ymm3, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+16]
    vfmadd231ps ymm4, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+20]
    vfmadd231ps ymm5, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+24]
    vfmadd231ps ymm6, ymm8, ymm9
    vbroadcastss ymm9, DWORD PTR [rax+28]
    vfmadd231ps ymm7, ymm8, ymm9
    add rdi, QWORD PTR [rsp+8]
    add rax, 32
    dec r8
    jnz .k_vec
    add r11, r14
    dec r9
    jnz .ci_vec

    lea r11, [rcx+r10*4]
    mov r9, r13
    shl r9, 2
    vmovups YMMWORD PTR [r11], ymm0
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm1
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm2
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm3
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm4
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm5
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm6
    add r11, r9
    vmovups YMMWORD PTR [r11], ymm7
    add r10, 8
    jmp .t8_check

.tail_check:
    cmp r10, r13
    jae .next_ob
.tail_one:
    vmovss xmm0, DWORD PTR [rdx+0]
    vmovss xmm1, DWORD PTR [rdx+4]
    vmovss xmm2, DWORD PTR [rdx+8]
    vmovss xmm3, DWORD PTR [rdx+12]
    vmovss xmm4, DWORD PTR [rdx+16]
    vmovss xmm5, DWORD PTR [rdx+20]
    vmovss xmm6, DWORD PTR [rdx+24]
    vmovss xmm7, DWORD PTR [rdx+28]

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    mov r9, r15
.ci_scalar:
    mov rdi, r11
    mov r8, r12
.k_scalar:
    vmovss xmm8, DWORD PTR [rdi]
    vfmadd231ss xmm0, xmm8, DWORD PTR [rax+0]
    vfmadd231ss xmm1, xmm8, DWORD PTR [rax+4]
    vfmadd231ss xmm2, xmm8, DWORD PTR [rax+8]
    vfmadd231ss xmm3, xmm8, DWORD PTR [rax+12]
    vfmadd231ss xmm4, xmm8, DWORD PTR [rax+16]
    vfmadd231ss xmm5, xmm8, DWORD PTR [rax+20]
    vfmadd231ss xmm6, xmm8, DWORD PTR [rax+24]
    vfmadd231ss xmm7, xmm8, DWORD PTR [rax+28]
    add rdi, QWORD PTR [rsp+8]
    add rax, 32
    dec r8
    jnz .k_scalar
    add r11, r14
    dec r9
    jnz .ci_scalar

    lea r11, [rcx+r10*4]
    mov r9, r13
    shl r9, 2
    vmovss DWORD PTR [r11], xmm0
    add r11, r9
    vmovss DWORD PTR [r11], xmm1
    add r11, r9
    vmovss DWORD PTR [r11], xmm2
    add r11, r9
    vmovss DWORD PTR [r11], xmm3
    add r11, r9
    vmovss DWORD PTR [r11], xmm4
    add r11, r9
    vmovss DWORD PTR [r11], xmm5
    add r11, r9
    vmovss DWORD PTR [r11], xmm6
    add r11, r9
    vmovss DWORD PTR [r11], xmm7
    inc r10
    cmp r10, r13
    jb .tail_one

.next_ob:
    add rsi, QWORD PTR [rsp+16]
    add rdx, 32
    mov rax, r13
    shl rax, 5                       # 8*Tout*sizeof(float)
    add rcx, rax
    dec rbx
    jnz .ob_loop

.done:
    vzeroupper
    add rsp, 40
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8,.-ds_conv1d_nct_f32_avx2_oc8_t8
.section .note.GNU-stack,"",@progbits
