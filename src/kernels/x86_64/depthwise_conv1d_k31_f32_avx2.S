.intel_syntax noprefix
.text

# void ds_depthwise_conv1d_k31_f32_avx2(
#     const float *x_padded, // rdi [C,T+30], 15 zeros on each side
#     const float *weight,   // rsi [C,31]
#     const float *bias,     // rdx [C]
#     float *y,              // rcx [C,T]
#     size_t C,              // r8
#     size_t T               // r9
# );
# Exact hot operator used by LYNXNet2Block before its Linear/GLU stack.
# Caller supplies padding so there are no boundary checks in the hot loop.

.globl ds_depthwise_conv1d_k31_f32_avx2
.type ds_depthwise_conv1d_k31_f32_avx2, @function
ds_depthwise_conv1d_k31_f32_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r8                         # C
    mov r13, r9                         # T
    test r12, r12
    jz .done
    test r13, r13
    jz .done

    mov r14, r13
    and r14, -8
    lea r15, [r13 + 30]
    shl r15, 2                          # input channel stride bytes
    mov r8, r13
    shl r8, 2                           # output channel stride bytes

.channel_loop:
    xor eax, eax
.t_vec:
    cmp rax, r14
    jae .t_vec_done
    vbroadcastss ymm0, DWORD PTR [rdx]
    lea r11, [rdi + rax*4]
    xor r10d, r10d
.tap_vec:
    vbroadcastss ymm1, DWORD PTR [rsi + r10*4]
    vmovups ymm2, YMMWORD PTR [r11 + r10*4]
    vfmadd231ps ymm0, ymm2, ymm1
    inc r10
    cmp r10, 31
    jb .tap_vec
    vmovups YMMWORD PTR [rcx + rax*4], ymm0
    add rax, 8
    jmp .t_vec

.t_vec_done:
    cmp rax, r13
    jae .channel_done
.t_scalar:
    vmovss xmm0, DWORD PTR [rdx]
    lea r11, [rdi + rax*4]
    xor r10d, r10d
.tap_scalar:
    vmovss xmm1, DWORD PTR [r11 + r10*4]
    vfmadd231ss xmm0, xmm1, DWORD PTR [rsi + r10*4]
    inc r10
    cmp r10, 31
    jb .tap_scalar
    vmovss DWORD PTR [rcx + rax*4], xmm0
    inc rax
    cmp rax, r13
    jb .t_scalar

.channel_done:
    add rdi, r15
    add rsi, 124
    add rdx, 4
    add rcx, r8
    dec r12
    jnz .channel_loop

.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_depthwise_conv1d_k31_f32_avx2, .-ds_depthwise_conv1d_k31_f32_avx2
.section .note.GNU-stack,"",@progbits
