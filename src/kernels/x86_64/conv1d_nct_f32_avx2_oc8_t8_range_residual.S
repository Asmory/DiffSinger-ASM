.intel_syntax noprefix
.text

# M38 range kernel with residual add fused into stores.
# Packed weights: [Cin,K,8] for exactly 8 output channels.
# void ds_conv1d_nct_f32_avx2_oc8_t8_range_residual(
#   const float *x_padded, const float *w_block, const float *bias8, float *y_block,
#   size_t Cin, size_t K, size_t t0, size_t tcount, size_t Tp,
#   size_t dilation, size_t y_stride);
.globl ds_conv1d_nct_f32_avx2_oc8_t8_range_residual
.type ds_conv1d_nct_f32_avx2_oc8_t8_range_residual,@function
ds_conv1d_nct_f32_avx2_oc8_t8_range_residual:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 40

    test r8, r8                    # Cin
    jz .done
    test r9, r9                    # K
    jz .done
    mov r12, r8                    # Cin
    mov r13, r9                    # K
    mov r14, QWORD PTR [rbp+32]    # Tp
    shl r14, 2                     # Tp bytes
    mov r15, QWORD PTR [rbp+48]    # y_stride
    shl r15, 2                     # bytes
    mov rax, QWORD PTR [rbp+40]    # dilation
    shl rax, 2
    mov QWORD PTR [rsp+0], rax     # dilation bytes
    mov rax, QWORD PTR [rbp+16]    # t0
    mov QWORD PTR [rsp+8], rax
    mov rbx, QWORD PTR [rbp+24]    # tcount
    add rbx, rax                   # tend
    mov QWORD PTR [rsp+16], rbx
    mov rax, rbx
    sub rax, QWORD PTR [rsp+8]
    and rax, -8
    add rax, QWORD PTR [rsp+8]
    mov QWORD PTR [rsp+24], rax    # vector end
    mov QWORD PTR [rsp+32], rdi    # original x

    mov r10, QWORD PTR [rsp+8]     # t
.vec_check:
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

    mov r11, QWORD PTR [rsp+32]
    lea r11, [r11+r10*4]
    mov rax, rsi
    mov r9, r12
.ci_vec:
    mov rdi, r11
    mov r8, r13
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
    add rdi, QWORD PTR [rsp+0]
    add rax, 32
    dec r8
    jnz .k_vec
    add r11, r14
    dec r9
    jnz .ci_vec

    lea r11, [rcx+r10*4]
    mov rax, QWORD PTR [rbp+56]
    lea rdi, [rax+r10*4]
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm0, ymm0, ymm8
    vmovups YMMWORD PTR [r11], ymm0
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm1, ymm1, ymm8
    vmovups YMMWORD PTR [r11], ymm1
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm2, ymm2, ymm8
    vmovups YMMWORD PTR [r11], ymm2
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm3, ymm3, ymm8
    vmovups YMMWORD PTR [r11], ymm3
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm4, ymm4, ymm8
    vmovups YMMWORD PTR [r11], ymm4
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm5, ymm5, ymm8
    vmovups YMMWORD PTR [r11], ymm5
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm6, ymm6, ymm8
    vmovups YMMWORD PTR [r11], ymm6
    add r11, r15
    add rdi, r15
    vmovups ymm8, YMMWORD PTR [rdi]
    vaddps ymm7, ymm7, ymm8
    vmovups YMMWORD PTR [r11], ymm7
    add r10, 8
    jmp .vec_check

.tail_check:
    cmp r10, QWORD PTR [rsp+16]
    jae .done
.tail_one:
    vmovss xmm0, DWORD PTR [rdx+0]
    vmovss xmm1, DWORD PTR [rdx+4]
    vmovss xmm2, DWORD PTR [rdx+8]
    vmovss xmm3, DWORD PTR [rdx+12]
    vmovss xmm4, DWORD PTR [rdx+16]
    vmovss xmm5, DWORD PTR [rdx+20]
    vmovss xmm6, DWORD PTR [rdx+24]
    vmovss xmm7, DWORD PTR [rdx+28]

    mov r11, QWORD PTR [rsp+32]
    lea r11, [r11+r10*4]
    mov rax, rsi
    mov r9, r12
.ci_scalar:
    mov rdi, r11
    mov r8, r13
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
    add rdi, QWORD PTR [rsp+0]
    add rax, 32
    dec r8
    jnz .k_scalar
    add r11, r14
    dec r9
    jnz .ci_scalar

    lea r11, [rcx+r10*4]
    mov rax, QWORD PTR [rbp+56]
    lea rdi, [rax+r10*4]
    vaddss xmm0, xmm0, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm0
    add r11, r15
    add rdi, r15
    vaddss xmm1, xmm1, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm1
    add r11, r15
    add rdi, r15
    vaddss xmm2, xmm2, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm2
    add r11, r15
    add rdi, r15
    vaddss xmm3, xmm3, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm3
    add r11, r15
    add rdi, r15
    vaddss xmm4, xmm4, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm4
    add r11, r15
    add rdi, r15
    vaddss xmm5, xmm5, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm5
    add r11, r15
    add rdi, r15
    vaddss xmm6, xmm6, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm6
    add r11, r15
    add rdi, r15
    vaddss xmm7, xmm7, DWORD PTR [rdi]
    vmovss DWORD PTR [r11], xmm7
    inc r10
    cmp r10, QWORD PTR [rsp+16]
    jb .tail_one

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
.size ds_conv1d_nct_f32_avx2_oc8_t8_range_residual,.-ds_conv1d_nct_f32_avx2_oc8_t8_range_residual
.section .note.GNU-stack,"",@progbits
