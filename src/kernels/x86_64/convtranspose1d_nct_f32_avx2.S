.intel_syntax noprefix
.text
# One-output-channel direct ConvTranspose1d kernel.
# Weight layout for this entry: [Cin,K] in original ONNX tap order.
# group=1, dilation=1, output_padding=0. Runtime parallelizes across Cout.
#
# void ds_convtranspose1d_oc_f32_avx2(
#   const float *x,      // rdi [Cin,Tin]
#   const float *w_oc,   // rsi [Cin,K]
#   float bias,          // xmm0
#   float *y,            // rdx [Tout]
#   size_t Cin,          // rcx
#   size_t K,            // r8
#   size_t Tin,          // r9
#   size_t Tout,         // [rbp+16]
#   size_t pad,          // [rbp+24]
#   size_t stride        // [rbp+32]
# );
.globl ds_convtranspose1d_oc_f32_avx2
.type ds_convtranspose1d_oc_f32_avx2,@function
ds_convtranspose1d_oc_f32_avx2:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    test rcx,rcx
    jz .done
    test r8,r8
    jz .done
    test r9,r9
    jz .done

    mov r10,QWORD PTR [rbp+16]      # Tout
    mov r11,QWORD PTR [rbp+24]      # pad
    mov r12,QWORD PTR [rbp+32]      # stride
    mov r13,rcx                     # Cin remaining
    mov r14,rdi                     # x row
    mov r15,rsi                     # w row

    # y[:] = bias
    vbroadcastss ymm1,xmm0
    xor ebx,ebx
    mov rax,r10
    and rax,-8
.init_vec:
    cmp rbx,rax
    jae .init_tail
    vmovups YMMWORD PTR [rdx+rbx*4],ymm1
    add rbx,8
    jmp .init_vec
.init_tail:
    cmp rbx,r10
    jae .ci_loop
    vmovss DWORD PTR [rdx+rbx*4],xmm0
    inc rbx
    jmp .init_tail

.ci_loop:
    xor ebx,ebx                     # t=0
.t_loop:
    cmp rbx,r9
    jae .next_ci
    mov rax,rbx
    imul rax,r12
    sub rax,r11                    # signed start=t*stride-pad

    xor ecx,ecx                    # kb=0
    test rax,rax
    jns .kb_ok
    mov rcx,rax
    neg rcx                        # kb=-start
    cmp rcx,r8
    jae .next_t
.kb_ok:
    mov rdi,r10
    sub rdi,rax                    # Tout-start
    cmp rdi,r8
    cmova rdi,r8                   # ke=min(K,Tout-start)
    cmp rdi,rcx
    jbe .next_t

    vbroadcastss ymm5,DWORD PTR [r14+rbx*4]
    mov rsi,rcx                    # q=kb
.q_vec:
    lea rcx,[rsi+8]
    cmp rcx,rdi
    ja .q_scalar
    lea rcx,[rax+rsi]              # out index
    vmovups ymm0,YMMWORD PTR [rdx+rcx*4]
    vfmadd231ps ymm0,ymm5,YMMWORD PTR [r15+rsi*4]
    vmovups YMMWORD PTR [rdx+rcx*4],ymm0
    add rsi,8
    jmp .q_vec
.q_scalar:
    cmp rsi,rdi
    jae .next_t
    lea rcx,[rax+rsi]
    vmovss xmm0,DWORD PTR [rdx+rcx*4]
    vfmadd231ss xmm0,xmm5,DWORD PTR [r15+rsi*4]
    vmovss DWORD PTR [rdx+rcx*4],xmm0
    inc rsi
    jmp .q_scalar
.next_t:
    inc rbx
    jmp .t_loop
.next_ci:
    lea r14,[r14+r9*4]
    lea r15,[r15+r8*4]
    dec r13
    jnz .ci_loop
.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_convtranspose1d_oc_f32_avx2,.-ds_convtranspose1d_oc_f32_avx2

# M34 specialized NSF-HiFiGAN ConvTranspose1d kernel for the late upsamplers:
#   stride=2, K=4, pad=1, dilation=1, output_padding=0.
# One output channel per call. Weight layout [Cin,4] in original ONNX tap order.
# Rather than vectorizing across K (which is only 4), vectorize across 8 input
# time positions. For output index pair n:
#   y[2n]   = b + sum_ci (x[n]*w1 + x[n-1]*w3)
#   y[2n+1] = b + sum_ci (x[n]*w2 + x[n+1]*w0)
# Boundary terms outside [0,Tin) are zero. Interior blocks compute 8 pairs at
# once and interleave even/odd YMM accumulators into 16 contiguous outputs.
#
# void ds_convtranspose1d_s2k4_oc_f32_avx2(
#   const float *x,      // rdi [Cin,Tin]
#   const float *w_oc,   // rsi [Cin,4]
#   float bias,          // xmm0
#   float *y,            // rdx [2*Tin]
#   size_t Cin,          // rcx
#   size_t Tin           // r8
# );
.globl ds_convtranspose1d_s2k4_oc_f32_avx2
.type ds_convtranspose1d_s2k4_oc_f32_avx2,@function
ds_convtranspose1d_s2k4_oc_f32_avx2:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp,16

    test rcx,rcx
    jz .s2_done
    test r8,r8
    jz .s2_done

    mov r12,rdi                    # x base
    mov r13,rsi                    # w base
    mov r14,rdx                    # y base
    mov r15,rcx                    # Cin
    mov rbx,r8                     # Tin
    mov r8,rbx
    shl r8,2                       # row_bytes = Tin*4
    vmovss DWORD PTR [rsp],xmm0    # saved bias

    # n=0 boundary pair.
    xor r9d,r9d
    call .s2_scalar_pair
    mov r9,1

.s2_vec_check:
    # Need n>=1 and n+8<Tin so prev/current/next 8-float loads are all valid.
    lea rax,[r9+8]
    cmp rax,rbx
    jae .s2_scalar_tail

    vbroadcastss ymm0,DWORD PTR [rsp]   # even accumulator
    vmovaps ymm1,ymm0                   # odd accumulator
    lea r10,[r12+r9*4]                  # x row 0 at n
    mov r11,r13                         # w row 0
    mov rcx,r15
.s2_vec_ci:
    vmovups ymm4,YMMWORD PTR [r10]      # x[n:n+8]
    vmovups ymm5,YMMWORD PTR [r10-4]    # x[n-1:n+7]
    vmovups ymm6,YMMWORD PTR [r10+4]    # x[n+1:n+9]

    vbroadcastss ymm7,DWORD PTR [r11+4] # w1
    vfmadd231ps ymm0,ymm4,ymm7
    vbroadcastss ymm7,DWORD PTR [r11+12]# w3
    vfmadd231ps ymm0,ymm5,ymm7

    vbroadcastss ymm7,DWORD PTR [r11+8] # w2
    vfmadd231ps ymm1,ymm4,ymm7
    vbroadcastss ymm7,DWORD PTR [r11]   # w0
    vfmadd231ps ymm1,ymm6,ymm7

    add r10,r8
    add r11,16
    dec rcx
    jnz .s2_vec_ci

    # Interleave even/odd [e0..e7], [o0..o7] -> 16 contiguous y samples.
    vunpcklps ymm2,ymm0,ymm1
    vunpckhps ymm3,ymm0,ymm1
    vperm2f128 ymm4,ymm2,ymm3,0x20
    vperm2f128 ymm5,ymm2,ymm3,0x31
    lea rax,[r9+r9]
    vmovups YMMWORD PTR [r14+rax*4],ymm4
    vmovups YMMWORD PTR [r14+rax*4+32],ymm5

    add r9,8
    jmp .s2_vec_check

.s2_scalar_tail:
    cmp r9,rbx
    jae .s2_done
.s2_scalar_loop:
    call .s2_scalar_pair
    inc r9
    cmp r9,rbx
    jb .s2_scalar_loop
    jmp .s2_done

# Internal scalar pair helper. r9=n, preserves r9/r12-r15/rbx/r8.
# Computes y[2n], y[2n+1] from all input channels.
.s2_scalar_pair:
    vmovss xmm0,DWORD PTR [rsp+8]   # even (call pushed return address)
    vmovss xmm1,DWORD PTR [rsp+8]   # odd
    lea r10,[r12+r9*4]
    mov r11,r13
    mov rcx,r15
.s2_scalar_ci:
    vmovss xmm4,DWORD PTR [r10]     # cur
    vmovss xmm5,DWORD PTR [r11+4]   # w1
    vfmadd231ss xmm0,xmm4,xmm5
    vmovss xmm5,DWORD PTR [r11+8]   # w2
    vfmadd231ss xmm1,xmm4,xmm5

    test r9,r9
    jz .s2_no_prev
    vmovss xmm4,DWORD PTR [r10-4]
    vmovss xmm5,DWORD PTR [r11+12]  # w3
    vfmadd231ss xmm0,xmm4,xmm5
.s2_no_prev:
    lea rax,[r9+1]
    cmp rax,rbx
    jae .s2_no_next
    vmovss xmm4,DWORD PTR [r10+4]
    vmovss xmm5,DWORD PTR [r11]     # w0
    vfmadd231ss xmm1,xmm4,xmm5
.s2_no_next:
    add r10,r8
    add r11,16
    dec rcx
    jnz .s2_scalar_ci

    lea rax,[r9+r9]
    vmovss DWORD PTR [r14+rax*4],xmm0
    vmovss DWORD PTR [r14+rax*4+4],xmm1
    ret

.s2_done:
    vzeroupper
    add rsp,16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_convtranspose1d_s2k4_oc_f32_avx2,.-ds_convtranspose1d_s2k4_oc_f32_avx2


# M42 specialized NSF-HiFiGAN early upsampler:
#   stride=8, K=16, pad=4, dilation=1, output_padding=0, Tout=8*Tin.
# One output channel per call. Weight layout [Cin,16] in original ONNX tap order.
#
# The generic kernel scatters each x[t] into sixteen y samples and therefore
# read/modify/writes the output once per input channel. Here we invert the loop:
# keep output samples in registers across the complete Cin reduction and store
# each y sample only once. For output block n (samples 8n..8n+7):
#   lo[0:4] = b + sum_ci x[n]  * w[4:8]  + x[n-1] * w[12:16]
#   hi[0:4] = b + sum_ci x[n]  * w[8:12] + x[n+1] * w[0:4]
# Four adjacent output blocks are computed together so each 16-float weight row
# is loaded once for 32 output samples.
#
# void ds_convtranspose1d_s8k16_oc_f32_avx2(
#   const float *x,      // rdi [Cin,Tin]
#   const float *w_oc,   // rsi [Cin,16]
#   float bias,          // xmm0
#   float *y,            // rdx [8*Tin]
#   size_t Cin,          // rcx
#   size_t Tin           // r8
# );
.globl ds_convtranspose1d_s8k16_oc_f32_avx2
.type ds_convtranspose1d_s8k16_oc_f32_avx2,@function
ds_convtranspose1d_s8k16_oc_f32_avx2:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp,16

    test rcx,rcx
    jz .s8_done
    test r8,r8
    jz .s8_done

    mov r12,rdi                    # x base
    mov r13,rsi                    # w base
    mov r14,rdx                    # y base
    mov r15,rcx                    # Cin
    mov rbx,r8                     # Tin
    vmovss DWORD PTR [rsp],xmm0    # bias
    mov rax,rbx
    shl rax,2
    mov QWORD PTR [rsp+8],rax      # input row bytes

    # n=0: no x[n-1] contribution.
    vbroadcastss xmm0,DWORD PTR [rsp]
    vmovaps xmm1,xmm0
    mov r10,r12
    mov r11,r13
    mov rcx,r15
.s8_first_ci:
    vmovups xmm8,XMMWORD PTR [r11]       # w0..3
    vmovups xmm9,XMMWORD PTR [r11+16]    # w4..7
    vmovups xmm10,XMMWORD PTR [r11+32]   # w8..11
    vbroadcastss xmm12,DWORD PTR [r10]
    vfmadd231ps xmm0,xmm12,xmm9
    vfmadd231ps xmm1,xmm12,xmm10
    cmp rbx,1
    jbe .s8_first_no_next
    vbroadcastss xmm12,DWORD PTR [r10+4]
    vfmadd231ps xmm1,xmm12,xmm8
.s8_first_no_next:
    add r10,QWORD PTR [rsp+8]
    add r11,64
    dec rcx
    jnz .s8_first_ci
    vmovups XMMWORD PTR [r14],xmm0
    vmovups XMMWORD PTR [r14+16],xmm1

    cmp rbx,1
    jbe .s8_done
    mov r9,1                      # next output block n

.s8_group_check:
    lea rax,[r9+4]
    cmp rax,rbx
    jae .s8_tail

    # Four interior blocks n..n+3. All prev/current/next accesses valid.
    vbroadcastss xmm0,DWORD PTR [rsp]
    vmovaps xmm1,xmm0
    vmovaps xmm2,xmm0
    vmovaps xmm3,xmm0
    vmovaps xmm4,xmm0
    vmovaps xmm5,xmm0
    vmovaps xmm6,xmm0
    vmovaps xmm7,xmm0

    lea r10,[r12+r9*4]
    mov r11,r13
    mov rcx,r15
.s8_group_ci:
    vmovups xmm8,XMMWORD PTR [r11]       # w0..3
    vmovups xmm9,XMMWORD PTR [r11+16]    # w4..7
    vmovups xmm10,XMMWORD PTR [r11+32]   # w8..11
    vmovups xmm11,XMMWORD PTR [r11+48]   # w12..15

    # x[n-1]
    vbroadcastss xmm12,DWORD PTR [r10-4]
    vfmadd231ps xmm0,xmm12,xmm11

    # x[n]
    vbroadcastss xmm12,DWORD PTR [r10]
    vfmadd231ps xmm0,xmm12,xmm9
    vfmadd231ps xmm1,xmm12,xmm10
    vfmadd231ps xmm2,xmm12,xmm11

    # x[n+1]
    vbroadcastss xmm12,DWORD PTR [r10+4]
    vfmadd231ps xmm1,xmm12,xmm8
    vfmadd231ps xmm2,xmm12,xmm9
    vfmadd231ps xmm3,xmm12,xmm10
    vfmadd231ps xmm4,xmm12,xmm11

    # x[n+2]
    vbroadcastss xmm12,DWORD PTR [r10+8]
    vfmadd231ps xmm3,xmm12,xmm8
    vfmadd231ps xmm4,xmm12,xmm9
    vfmadd231ps xmm5,xmm12,xmm10
    vfmadd231ps xmm6,xmm12,xmm11

    # x[n+3]
    vbroadcastss xmm12,DWORD PTR [r10+12]
    vfmadd231ps xmm5,xmm12,xmm8
    vfmadd231ps xmm6,xmm12,xmm9
    vfmadd231ps xmm7,xmm12,xmm10

    # x[n+4]
    vbroadcastss xmm12,DWORD PTR [r10+16]
    vfmadd231ps xmm7,xmm12,xmm8

    add r10,QWORD PTR [rsp+8]
    add r11,64
    dec rcx
    jnz .s8_group_ci

    mov rax,r9
    shl rax,5                      # byte offset = n*8 floats*4
    vmovups XMMWORD PTR [r14+rax],xmm0
    vmovups XMMWORD PTR [r14+rax+16],xmm1
    vmovups XMMWORD PTR [r14+rax+32],xmm2
    vmovups XMMWORD PTR [r14+rax+48],xmm3
    vmovups XMMWORD PTR [r14+rax+64],xmm4
    vmovups XMMWORD PTR [r14+rax+80],xmm5
    vmovups XMMWORD PTR [r14+rax+96],xmm6
    vmovups XMMWORD PTR [r14+rax+112],xmm7
    add r9,4
    jmp .s8_group_check

.s8_tail:
    cmp r9,rbx
    jae .s8_done
    lea rax,[r9+1]
    cmp rax,rbx
    jae .s8_last

    # One interior block.
    vbroadcastss xmm0,DWORD PTR [rsp]
    vmovaps xmm1,xmm0
    lea r10,[r12+r9*4]
    mov r11,r13
    mov rcx,r15
.s8_single_ci:
    vmovups xmm8,XMMWORD PTR [r11]
    vmovups xmm9,XMMWORD PTR [r11+16]
    vmovups xmm10,XMMWORD PTR [r11+32]
    vmovups xmm11,XMMWORD PTR [r11+48]
    vbroadcastss xmm12,DWORD PTR [r10]
    vfmadd231ps xmm0,xmm12,xmm9
    vfmadd231ps xmm1,xmm12,xmm10
    vbroadcastss xmm12,DWORD PTR [r10-4]
    vfmadd231ps xmm0,xmm12,xmm11
    vbroadcastss xmm12,DWORD PTR [r10+4]
    vfmadd231ps xmm1,xmm12,xmm8
    add r10,QWORD PTR [rsp+8]
    add r11,64
    dec rcx
    jnz .s8_single_ci
    mov rax,r9
    shl rax,5
    vmovups XMMWORD PTR [r14+rax],xmm0
    vmovups XMMWORD PTR [r14+rax+16],xmm1
    inc r9
    jmp .s8_tail

.s8_last:
    # n=Tin-1: no x[n+1] contribution.
    vbroadcastss xmm0,DWORD PTR [rsp]
    vmovaps xmm1,xmm0
    lea r10,[r12+r9*4]
    mov r11,r13
    mov rcx,r15
.s8_last_ci:
    vmovups xmm9,XMMWORD PTR [r11+16]
    vmovups xmm10,XMMWORD PTR [r11+32]
    vmovups xmm11,XMMWORD PTR [r11+48]
    vbroadcastss xmm12,DWORD PTR [r10]
    vfmadd231ps xmm0,xmm12,xmm9
    vfmadd231ps xmm1,xmm12,xmm10
    vbroadcastss xmm12,DWORD PTR [r10-4]
    vfmadd231ps xmm0,xmm12,xmm11
    add r10,QWORD PTR [rsp+8]
    add r11,64
    dec rcx
    jnz .s8_last_ci
    mov rax,r9
    shl rax,5
    vmovups XMMWORD PTR [r14+rax],xmm0
    vmovups XMMWORD PTR [r14+rax+16],xmm1

.s8_done:
    vzeroupper
    add rsp,16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_convtranspose1d_s8k16_oc_f32_avx2,.-ds_convtranspose1d_s8k16_oc_f32_avx2

.section .note.GNU-stack,"",@progbits
