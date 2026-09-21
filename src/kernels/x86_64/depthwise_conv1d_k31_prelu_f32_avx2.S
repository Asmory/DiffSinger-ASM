.intel_syntax noprefix
.text

# void ds_depthwise_conv1d_k31_prelu_f32_avx2(
#     const float *x_padded,   // rdi: [C, T+30], zero padded 15 left/right
#     const float *weight,     // rsi: [C, 31]
#     const float *bias,       // rdx: [C]
#     const float *slope,      // rcx: [C] PReLU slope
#     float *y,                // r8 : [C, T]
#     size_t C,                // r9
#     size_t T                 // [rbp+16]
# );
#
# Fuses the exact LYNXConvModule hot path:
#   depthwise Conv1d(kernel=31, padding=15, groups=C) -> channel PReLU
#
# The caller/converter runtime supplies x_padded so the hot loop contains no
# boundary branches. System V AMD64 ABI. AVX2 + FMA3 required.

.globl ds_depthwise_conv1d_k31_prelu_f32_avx2
.type ds_depthwise_conv1d_k31_prelu_f32_avx2, @function
ds_depthwise_conv1d_k31_prelu_f32_avx2:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r9                         # channels remaining
    mov r13, QWORD PTR [rbp + 16]       # T
    test r12, r12
    jz .done
    test r13, r13
    jz .done

    mov r14, r13
    and r14, -8                         # vectorized T end

    lea r15, [r13 + 30]
    shl r15, 2                          # x padded channel stride bytes

    mov r9, r13
    shl r9, 2                           # y channel stride bytes

.channel_loop:
    xor eax, eax                        # t = 0

.t_vec_loop:
    cmp rax, r14
    jae .t_vec_done

    vbroadcastss ymm0, DWORD PTR [rdx]  # acc = bias[c]
    lea r11, [rdi + rax*4]              # x_padded[c, t]
    xor r10d, r10d                      # tap = 0

.tap_vec_loop:
    vbroadcastss ymm1, DWORD PTR [rsi + r10*4]
    vmovups ymm2, YMMWORD PTR [r11 + r10*4]
    vfmadd231ps ymm0, ymm2, ymm1
    inc r10
    cmp r10, 31
    jb .tap_vec_loop

    # PReLU(v,a) = max(v,0) + a * min(v,0)
    vxorps ymm3, ymm3, ymm3
    vmaxps ymm4, ymm0, ymm3
    vminps ymm5, ymm0, ymm3
    vbroadcastss ymm6, DWORD PTR [rcx]
    vfmadd231ps ymm4, ymm5, ymm6
    vmovups YMMWORD PTR [r8 + rax*4], ymm4

    add rax, 8
    jmp .t_vec_loop

.t_vec_done:
    cmp rax, r13
    jae .channel_done

.t_scalar_loop:
    vmovss xmm0, DWORD PTR [rdx]        # acc = bias[c]
    lea r11, [rdi + rax*4]
    xor r10d, r10d

.tap_scalar_loop:
    vmovss xmm1, DWORD PTR [r11 + r10*4]
    vfmadd231ss xmm0, xmm1, DWORD PTR [rsi + r10*4]
    inc r10
    cmp r10, 31
    jb .tap_scalar_loop

    vxorps xmm2, xmm2, xmm2
    vmaxss xmm3, xmm0, xmm2
    vminss xmm4, xmm0, xmm2
    vmulss xmm4, xmm4, DWORD PTR [rcx]
    vaddss xmm3, xmm3, xmm4
    vmovss DWORD PTR [r8 + rax*4], xmm3

    inc rax
    cmp rax, r13
    jb .t_scalar_loop

.channel_done:
    add rdi, r15                       # next padded input channel
    add rsi, 124                       # 31 * sizeof(float)
    add rdx, 4
    add rcx, 4
    add r8, r9

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
.size ds_depthwise_conv1d_k31_prelu_f32_avx2, .-ds_depthwise_conv1d_k31_prelu_f32_avx2

.section .note.GNU-stack,"",@progbits
