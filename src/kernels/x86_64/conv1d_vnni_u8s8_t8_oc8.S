.intel_syntax noprefix
.text
# M39 AVX-VNNI feasibility kernel.
# xpack: [Tblocks,K4,8 lanes x 4 u8 reductions] = 32 bytes per K4.
# wpack: [Cout/8,K4,8 output channels x 4 s8 weights] = 32 bytes per K4.
# corr[oc] = -128 * sum(qw), scale[oc] = activation_scale * weight_scale[oc].
# void ds_vnni_conv1d_u8s8_t8_oc8(
#   const uint8_t *xpack,const int8_t *wpack,const int32_t *corr,
#   const float *scale,const float *bias,float *y,
#   size_t Tblocks,size_t K4,size_t Cout,size_t Tout);
.globl ds_vnni_conv1d_u8s8_t8_oc8
.type ds_vnni_conv1d_u8s8_t8_oc8,@function
ds_vnni_conv1d_u8s8_t8_oc8:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    mov r12, QWORD PTR [rbp+16]      # Tblocks
    mov r13, QWORD PTR [rbp+24]      # K4
    mov r14, QWORD PTR [rbp+32]      # Cout
    mov r15, QWORD PTR [rbp+40]      # Tout
    test r12,r12
    jz .done
    test r13,r13
    jz .done
    test r14,r14
    jz .done
    test r14,7
    jnz .done
    mov QWORD PTR [rsp+0], rdi       # xpack base
    mov rbx, r14
    shr rbx,3                        # output blocks

.ob:
    mov rdi, QWORD PTR [rsp+0]       # reset xpack for each output block
    xor r10d,r10d                    # tb
.tb:
    cmp r10,r12
    jae .next_ob
    vpxor ymm0,ymm0,ymm0
    vpxor ymm1,ymm1,ymm1
    vpxor ymm2,ymm2,ymm2
    vpxor ymm3,ymm3,ymm3
    vpxor ymm4,ymm4,ymm4
    vpxor ymm5,ymm5,ymm5
    vpxor ymm6,ymm6,ymm6
    vpxor ymm7,ymm7,ymm7
    mov rax,rsi                     # weight block start
    mov r11,r13
.k4:
    vmovdqu ymm8,YMMWORD PTR [rdi]
    vpbroadcastd ymm9,DWORD PTR [rax+0]
    {vex} vpdpbusd ymm0,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+4]
    {vex} vpdpbusd ymm1,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+8]
    {vex} vpdpbusd ymm2,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+12]
    {vex} vpdpbusd ymm3,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+16]
    {vex} vpdpbusd ymm4,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+20]
    {vex} vpdpbusd ymm5,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+24]
    {vex} vpdpbusd ymm6,ymm8,ymm9
    vpbroadcastd ymm9,DWORD PTR [rax+28]
    {vex} vpdpbusd ymm7,ymm8,ymm9
    add rdi,32
    add rax,32
    dec r11
    jnz .k4

    # integer zero-point correction, dequantize, bias, NCT store.
    mov rax,r10
    shl rax,5                       # tb*8 floats = tb*32 bytes
    add rax,r9                      # y for oc0,tb

    vpbroadcastd ymm8,DWORD PTR [rdx+0]
    vpaddd ymm0,ymm0,ymm8
    vcvtdq2ps ymm0,ymm0
    vbroadcastss ymm8,DWORD PTR [rcx+0]
    vmulps ymm0,ymm0,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+0]
    vaddps ymm0,ymm0,ymm8
    vmovups YMMWORD PTR [rax],ymm0
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+4]
    vpaddd ymm1,ymm1,ymm8
    vcvtdq2ps ymm1,ymm1
    vbroadcastss ymm8,DWORD PTR [rcx+4]
    vmulps ymm1,ymm1,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+4]
    vaddps ymm1,ymm1,ymm8
    vmovups YMMWORD PTR [rax],ymm1
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+8]
    vpaddd ymm2,ymm2,ymm8
    vcvtdq2ps ymm2,ymm2
    vbroadcastss ymm8,DWORD PTR [rcx+8]
    vmulps ymm2,ymm2,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+8]
    vaddps ymm2,ymm2,ymm8
    vmovups YMMWORD PTR [rax],ymm2
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+12]
    vpaddd ymm3,ymm3,ymm8
    vcvtdq2ps ymm3,ymm3
    vbroadcastss ymm8,DWORD PTR [rcx+12]
    vmulps ymm3,ymm3,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+12]
    vaddps ymm3,ymm3,ymm8
    vmovups YMMWORD PTR [rax],ymm3
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+16]
    vpaddd ymm4,ymm4,ymm8
    vcvtdq2ps ymm4,ymm4
    vbroadcastss ymm8,DWORD PTR [rcx+16]
    vmulps ymm4,ymm4,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+16]
    vaddps ymm4,ymm4,ymm8
    vmovups YMMWORD PTR [rax],ymm4
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+20]
    vpaddd ymm5,ymm5,ymm8
    vcvtdq2ps ymm5,ymm5
    vbroadcastss ymm8,DWORD PTR [rcx+20]
    vmulps ymm5,ymm5,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+20]
    vaddps ymm5,ymm5,ymm8
    vmovups YMMWORD PTR [rax],ymm5
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+24]
    vpaddd ymm6,ymm6,ymm8
    vcvtdq2ps ymm6,ymm6
    vbroadcastss ymm8,DWORD PTR [rcx+24]
    vmulps ymm6,ymm6,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+24]
    vaddps ymm6,ymm6,ymm8
    vmovups YMMWORD PTR [rax],ymm6
    lea rax,[rax+r15*4]

    vpbroadcastd ymm8,DWORD PTR [rdx+28]
    vpaddd ymm7,ymm7,ymm8
    vcvtdq2ps ymm7,ymm7
    vbroadcastss ymm8,DWORD PTR [rcx+28]
    vmulps ymm7,ymm7,ymm8
    vbroadcastss ymm8,DWORD PTR [r8+28]
    vaddps ymm7,ymm7,ymm8
    vmovups YMMWORD PTR [rax],ymm7

    inc r10
    jmp .tb

.next_ob:
    mov rax,r13
    shl rax,5                       # K4*32 bytes
    add rsi,rax
    add rdx,32
    add rcx,32
    add r8,32
    mov rax,r15
    shl rax,5                       # 8*Tout*4 bytes
    add r9,rax
    dec rbx
    jnz .ob
.done:
    vzeroupper
    add rsp,16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_vnni_conv1d_u8s8_t8_oc8,.-ds_vnni_conv1d_u8s8_t8_oc8
.section .note.GNU-stack,"",@progbits
