.intel_syntax noprefix
.text


# M39 fixed-K=3 Conv1d. Real HiFi-GAN T is 8-aligned; caller dispatches only then.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_k3
.type ds_conv1d_nct_f32_avx2_oc8_t8_k3,@function
ds_conv1d_nct_f32_avx2_oc8_t8_k3:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    test r8, r8
    jz .done_3
    test r9, r9
    jz .done_3
    test r9, 7
    jnz .done_3

    mov r12, r8                      # Cin
    mov r13, QWORD PTR [rbp+24]      # Tout
    test r13, 7
    jnz .done_3
    mov r14, QWORD PTR [rbp+32]      # Tp
    shl r14, 2                       # Tp bytes
    mov r15, QWORD PTR [rbp+40]      # dilation
    shl r15, 2                       # dilation bytes
    mov QWORD PTR [rsp+0], rdi       # original x
    mov rax, r12
    imul rax, 96
    mov QWORD PTR [rsp+8], rax       # Cin*K*8 floats bytes
    mov rbx, r9
    shr rbx, 3

.ob_loop_3:
    xor r10d, r10d
.t_loop_3:
    cmp r10, r13
    jae .next_ob_3

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
    mov r9, r12
.ci_3:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r14
    dec r9
    jnz .ci_3

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
    jmp .t_loop_3

.next_ob_3:
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .ob_loop_3

.done_3:
    vzeroupper
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_k3,.-ds_conv1d_nct_f32_avx2_oc8_t8_k3

# M39 fixed-K=3 one oc8 block over an 8-aligned time range.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_range_k3
.type ds_conv1d_nct_f32_avx2_oc8_t8_range_k3,@function
ds_conv1d_nct_f32_avx2_oc8_t8_range_k3:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .rdone_3
    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+32]       # Tp
    shl r13, 2
    mov r14, QWORD PTR [rbp+48]       # y_stride
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov r10, QWORD PTR [rbp+16]       # t0
    mov rbx, QWORD PTR [rbp+24]       # tcount
    test r10, 7
    jnz .rdone_3
    test rbx, 7
    jnz .rdone_3
    add rbx, r10                      # tend
    mov QWORD PTR [rsp+0], rdi        # original x

.rt_loop_3:
    cmp r10, rbx
    jae .rdone_3

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
    mov r9, r12
.rci_3:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r13
    dec r9
    jnz .rci_3

    lea r11, [rcx+r10*4]
    vmovups YMMWORD PTR [r11], ymm0
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm1
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm2
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm3
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm4
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm5
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm6
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm7
    add r10, 8
    jmp .rt_loop_3

.rdone_3:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_range_k3,.-ds_conv1d_nct_f32_avx2_oc8_t8_range_k3

# M39 fixed-K=7 Conv1d. Real HiFi-GAN T is 8-aligned; caller dispatches only then.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_k7
.type ds_conv1d_nct_f32_avx2_oc8_t8_k7,@function
ds_conv1d_nct_f32_avx2_oc8_t8_k7:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    test r8, r8
    jz .done_7
    test r9, r9
    jz .done_7
    test r9, 7
    jnz .done_7

    mov r12, r8                      # Cin
    mov r13, QWORD PTR [rbp+24]      # Tout
    test r13, 7
    jnz .done_7
    mov r14, QWORD PTR [rbp+32]      # Tp
    shl r14, 2                       # Tp bytes
    mov r15, QWORD PTR [rbp+40]      # dilation
    shl r15, 2                       # dilation bytes
    mov QWORD PTR [rsp+0], rdi       # original x
    mov rax, r12
    imul rax, 224
    mov QWORD PTR [rsp+8], rax       # Cin*K*8 floats bytes
    mov rbx, r9
    shr rbx, 3

.ob_loop_7:
    xor r10d, r10d
.t_loop_7:
    cmp r10, r13
    jae .next_ob_7

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
    mov r9, r12
.ci_7:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r14
    dec r9
    jnz .ci_7

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
    jmp .t_loop_7

.next_ob_7:
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .ob_loop_7

.done_7:
    vzeroupper
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_k7,.-ds_conv1d_nct_f32_avx2_oc8_t8_k7

# M39 fixed-K=7 one oc8 block over an 8-aligned time range.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_range_k7
.type ds_conv1d_nct_f32_avx2_oc8_t8_range_k7,@function
ds_conv1d_nct_f32_avx2_oc8_t8_range_k7:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .rdone_7
    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+32]       # Tp
    shl r13, 2
    mov r14, QWORD PTR [rbp+48]       # y_stride
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov r10, QWORD PTR [rbp+16]       # t0
    mov rbx, QWORD PTR [rbp+24]       # tcount
    test r10, 7
    jnz .rdone_7
    test rbx, 7
    jnz .rdone_7
    add rbx, r10                      # tend
    mov QWORD PTR [rsp+0], rdi        # original x

.rt_loop_7:
    cmp r10, rbx
    jae .rdone_7

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
    mov r9, r12
.rci_7:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r13
    dec r9
    jnz .rci_7

    lea r11, [rcx+r10*4]
    vmovups YMMWORD PTR [r11], ymm0
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm1
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm2
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm3
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm4
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm5
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm6
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm7
    add r10, 8
    jmp .rt_loop_7

.rdone_7:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_range_k7,.-ds_conv1d_nct_f32_avx2_oc8_t8_range_k7

# M39 fixed-K=11 Conv1d. Real HiFi-GAN T is 8-aligned; caller dispatches only then.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_k11
.type ds_conv1d_nct_f32_avx2_oc8_t8_k11,@function
ds_conv1d_nct_f32_avx2_oc8_t8_k11:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 16

    test r8, r8
    jz .done_11
    test r9, r9
    jz .done_11
    test r9, 7
    jnz .done_11

    mov r12, r8                      # Cin
    mov r13, QWORD PTR [rbp+24]      # Tout
    test r13, 7
    jnz .done_11
    mov r14, QWORD PTR [rbp+32]      # Tp
    shl r14, 2                       # Tp bytes
    mov r15, QWORD PTR [rbp+40]      # dilation
    shl r15, 2                       # dilation bytes
    mov QWORD PTR [rsp+0], rdi       # original x
    mov rax, r12
    imul rax, 352
    mov QWORD PTR [rsp+8], rax       # Cin*K*8 floats bytes
    mov rbx, r9
    shr rbx, 3

.ob_loop_11:
    xor r10d, r10d
.t_loop_11:
    cmp r10, r13
    jae .next_ob_11

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
    mov r9, r12
.ci_11:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r14
    dec r9
    jnz .ci_11

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
    jmp .t_loop_11

.next_ob_11:
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .ob_loop_11

.done_11:
    vzeroupper
    add rsp, 16
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_k11,.-ds_conv1d_nct_f32_avx2_oc8_t8_k11

# M39 fixed-K=11 one oc8 block over an 8-aligned time range.
.globl ds_conv1d_nct_f32_avx2_oc8_t8_range_k11
.type ds_conv1d_nct_f32_avx2_oc8_t8_range_k11,@function
ds_conv1d_nct_f32_avx2_oc8_t8_range_k11:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .rdone_11
    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+32]       # Tp
    shl r13, 2
    mov r14, QWORD PTR [rbp+48]       # y_stride
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov r10, QWORD PTR [rbp+16]       # t0
    mov rbx, QWORD PTR [rbp+24]       # tcount
    test r10, 7
    jnz .rdone_11
    test rbx, 7
    jnz .rdone_11
    add rbx, r10                      # tend
    mov QWORD PTR [rsp+0], rdi        # original x

.rt_loop_11:
    cmp r10, rbx
    jae .rdone_11

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
    mov r9, r12
.rci_11:
    mov rdi, r11
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
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
    add rdi, r15
    add rax, 32
    add r11, r13
    dec r9
    jnz .rci_11

    lea r11, [rcx+r10*4]
    vmovups YMMWORD PTR [r11], ymm0
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm1
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm2
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm3
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm4
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm5
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm6
    add r11, r14
    vmovups YMMWORD PTR [r11], ymm7
    add r10, 8
    jmp .rt_loop_11

.rdone_11:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc8_t8_range_k11,.-ds_conv1d_nct_f32_avx2_oc8_t8_range_k11


# M45 perf-guided K=7 kernel: oc4 x t24.
#
# M45.1 perf showed the K7 fixed-size inner loop dominated P-core cycles and
# that memory-source vbroadcastss instructions were the hottest individual
# instructions.  The old oc8xt8 kernel re-broadcasts the same eight weights for
# every t8 tile.  This kernel trades cheap L1 input reloads for weight-broadcast
# reuse: four output channels x three t8 vectors keep 12 accumulators live, so
# each scalar weight broadcast feeds 24 output samples before the next weight.
# Packed weights remain the existing [oc8-block][Cin][K][8] format and the
# accumulation order for every output element is unchanged.
.macro M45_K7_STEP
    vmovups ymm12, YMMWORD PTR [rdi]
    vmovups ymm13, YMMWORD PTR [rdi+32]
    vmovups ymm14, YMMWORD PTR [rdi+64]

    vbroadcastss ymm15, DWORD PTR [rax+0]
    vfmadd231ps ymm0, ymm12, ymm15
    vfmadd231ps ymm1, ymm13, ymm15
    vfmadd231ps ymm2, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+4]
    vfmadd231ps ymm3, ymm12, ymm15
    vfmadd231ps ymm4, ymm13, ymm15
    vfmadd231ps ymm5, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+8]
    vfmadd231ps ymm6, ymm12, ymm15
    vfmadd231ps ymm7, ymm13, ymm15
    vfmadd231ps ymm8, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+12]
    vfmadd231ps ymm9, ymm12, ymm15
    vfmadd231ps ymm10, ymm13, ymm15
    vfmadd231ps ymm11, ymm14, ymm15

    add rdi, r15
    add rax, 32
.endm

.globl ds_conv1d_nct_f32_avx2_oc4_t24_k7
.type ds_conv1d_nct_f32_avx2_oc4_t24_k7,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k7:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m45_done_7
    test r9, r9
    jz .m45_done_7
    test r9, 7
    jnz .m45_done_7

    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+24]       # Tout (caller gates Tout % 24 == 0)
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2                        # Tp bytes
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2                        # dilation bytes
    mov QWORD PTR [rsp+0], rdi        # original x
    mov rax, r12
    imul rax, 224                     # Cin * 7 * 8 * sizeof(float)
    mov QWORD PTR [rsp+8], rax        # packed-weight bytes / oc8 block
    mov rbx, r9
    shr rbx, 3                        # oc8 block count

.m45_ob_7:
    xor r8d, r8d                      # byte offset of lane 0 or lane 4 in packed8
.m45_half_7:
    # y-half base: lane4 starts four NCT channels later = 16*Tout bytes.
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi

    xor r10d, r10d
.m45_t_7:
    cmp r10, r13
    jae .m45_next_half_7

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m45_ci_7:
    mov rdi, r11
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    add r11, r14
    dec r9
    jnz .m45_ci_7

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2                        # one output-channel stride in bytes

    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m45_t_7

.m45_next_half_7:
    add r8, 16
    cmp r8, 32
    jb .m45_half_7

    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5                        # 8 * Tout * sizeof(float)
    add rcx, rax
    dec rbx
    jnz .m45_ob_7

.m45_done_7:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k7,.-ds_conv1d_nct_f32_avx2_oc4_t24_k7



# M46 perf-guided K=11 kernel: oc4 x t24.
# Same register-reuse strategy as M45 K7: each scalar packed weight is
# broadcast once and reused across three t8 time vectors for four output
# channels.  Packed8 layout and per-output accumulation order are unchanged.
.macro M46_K11_STEP
    vmovups ymm12, YMMWORD PTR [rdi]
    vmovups ymm13, YMMWORD PTR [rdi+32]
    vmovups ymm14, YMMWORD PTR [rdi+64]

    vbroadcastss ymm15, DWORD PTR [rax+0]
    vfmadd231ps ymm0, ymm12, ymm15
    vfmadd231ps ymm1, ymm13, ymm15
    vfmadd231ps ymm2, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+4]
    vfmadd231ps ymm3, ymm12, ymm15
    vfmadd231ps ymm4, ymm13, ymm15
    vfmadd231ps ymm5, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+8]
    vfmadd231ps ymm6, ymm12, ymm15
    vfmadd231ps ymm7, ymm13, ymm15
    vfmadd231ps ymm8, ymm14, ymm15

    vbroadcastss ymm15, DWORD PTR [rax+12]
    vfmadd231ps ymm9, ymm12, ymm15
    vfmadd231ps ymm10, ymm13, ymm15
    vfmadd231ps ymm11, ymm14, ymm15

    add rdi, r15
    add rax, 32
.endm

.globl ds_conv1d_nct_f32_avx2_oc4_t24_k11
.type ds_conv1d_nct_f32_avx2_oc4_t24_k11,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k11:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m46_done_11
    test r9, r9
    jz .m46_done_11
    test r9, 7
    jnz .m46_done_11

    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+24]       # Tout (caller gates Tout % 24 == 0)
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2                        # Tp bytes
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2                        # dilation bytes
    mov QWORD PTR [rsp+0], rdi        # original x
    mov rax, r12
    imul rax, 352                     # Cin * 11 * 8 * sizeof(float)
    mov QWORD PTR [rsp+8], rax        # packed-weight bytes / oc8 block
    mov rbx, r9
    shr rbx, 3                        # oc8 block count

.m46_ob_11:
    xor r8d, r8d                      # byte offset of lane 0 or lane 4 in packed8
.m46_half_11:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi

    xor r10d, r10d
.m46_t_11:
    cmp r10, r13
    jae .m46_next_half_11

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m46_ci_11:
    mov rdi, r11
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    add r11, r14
    dec r9
    jnz .m46_ci_11

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2

    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m46_t_11

.m46_next_half_11:
    add r8, 16
    cmp r8, 32
    jb .m46_half_11

    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .m46_ob_11

.m46_done_11:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k11,.-ds_conv1d_nct_f32_avx2_oc4_t24_k11


# M47: 2-D/range variants of the proven oc4 x t24 K7/K11 kernels.
# Signature matches the existing range kernels:
#   x,w,b,y,Cin,K,t0,tcount,Tp,dilation,y_stride
# Each call owns one packed8 output block and computes two oc4 halves.
.globl ds_conv1d_nct_f32_avx2_oc4_t24_range_k7
.type ds_conv1d_nct_f32_avx2_oc4_t24_range_k7,@function
ds_conv1d_nct_f32_avx2_oc4_t24_range_k7:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m47_rdone_7
    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+32]       # Tp
    shl r13, 2                        # Tp bytes
    mov r14, QWORD PTR [rbp+48]       # y_stride
    shl r14, 2                        # y_stride bytes
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2                        # dilation bytes
    mov r10, QWORD PTR [rbp+16]       # t0
    mov rbx, QWORD PTR [rbp+24]       # tcount
    test rbx, rbx
    jz .m47_rdone_7
    mov QWORD PTR [rsp+0], rdi        # original x
    mov QWORD PTR [rsp+8], r10        # t0
    add rbx, r10
    mov QWORD PTR [rsp+16], rbx       # tend

    xor r8d, r8d                      # packed-lane byte offset: 0 / 16
.m47_rhalf_7:
    mov r10, QWORD PTR [rsp+8]
.m47_rt_7:
    cmp r10, QWORD PTR [rsp+16]
    jae .m47_rnext_half_7

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m47_rci_7:
    mov rdi, r11
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    add r11, r13
    dec r9
    jnz .m47_rci_7

    mov rdi, r8
    shr rdi, 2                        # 0 or 4 output channels
    imul rdi, r14                     # byte offset to first channel of half
    add rdi, rcx
    lea rdi, [rdi+r10*4]

    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m47_rt_7

.m47_rnext_half_7:
    add r8, 16
    cmp r8, 32
    jb .m47_rhalf_7

.m47_rdone_7:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_range_k7,.-ds_conv1d_nct_f32_avx2_oc4_t24_range_k7

.globl ds_conv1d_nct_f32_avx2_oc4_t24_range_k11
.type ds_conv1d_nct_f32_avx2_oc4_t24_range_k11,@function
ds_conv1d_nct_f32_avx2_oc4_t24_range_k11:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m47_rdone_11
    mov r12, r8                       # Cin
    mov r13, QWORD PTR [rbp+32]       # Tp
    shl r13, 2
    mov r14, QWORD PTR [rbp+48]       # y_stride
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov r10, QWORD PTR [rbp+16]       # t0
    mov rbx, QWORD PTR [rbp+24]       # tcount
    test rbx, rbx
    jz .m47_rdone_11
    mov QWORD PTR [rsp+0], rdi
    mov QWORD PTR [rsp+8], r10
    add rbx, r10
    mov QWORD PTR [rsp+16], rbx

    xor r8d, r8d
.m47_rhalf_11:
    mov r10, QWORD PTR [rsp+8]
.m47_rt_11:
    cmp r10, QWORD PTR [rsp+16]
    jae .m47_rnext_half_11

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m47_rci_11:
    mov rdi, r11
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    add r11, r13
    dec r9
    jnz .m47_rci_11

    mov rdi, r8
    shr rdi, 2
    imul rdi, r14
    add rdi, rcx
    lea rdi, [rdi+r10*4]

    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r14
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m47_rt_11

.m47_rnext_half_11:
    add r8, 16
    cmp r8, 32
    jb .m47_rhalf_11

.m47_rdone_11:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_range_k11,.-ds_conv1d_nct_f32_avx2_oc4_t24_range_k11

.section .note.GNU-stack,"",@progbits
.text

# M48 perf-guided K=3 kernels. Compare oc4 x t16 vs oc4 x t24.
# Both preserve packed8 weights and per-output accumulation order.
.macro M48_K3_T16_STEP
    vmovups ymm8, YMMWORD PTR [rdi]
    vmovups ymm9, YMMWORD PTR [rdi+32]

    vbroadcastss ymm10, DWORD PTR [rax+0]
    vfmadd231ps ymm0, ymm8, ymm10
    vfmadd231ps ymm1, ymm9, ymm10
    vbroadcastss ymm10, DWORD PTR [rax+4]
    vfmadd231ps ymm2, ymm8, ymm10
    vfmadd231ps ymm3, ymm9, ymm10
    vbroadcastss ymm10, DWORD PTR [rax+8]
    vfmadd231ps ymm4, ymm8, ymm10
    vfmadd231ps ymm5, ymm9, ymm10
    vbroadcastss ymm10, DWORD PTR [rax+12]
    vfmadd231ps ymm6, ymm8, ymm10
    vfmadd231ps ymm7, ymm9, ymm10

    add rdi, r15
    add rax, 32
.endm

.globl ds_conv1d_nct_f32_avx2_oc4_t16_k3
.type ds_conv1d_nct_f32_avx2_oc4_t16_k3,@function
ds_conv1d_nct_f32_avx2_oc4_t16_k3:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m48_done_3_t16
    test r9, r9
    jz .m48_done_3_t16
    test r9, 7
    jnz .m48_done_3_t16

    mov r12, r8
    mov r13, QWORD PTR [rbp+24]       # Tout
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov QWORD PTR [rsp+0], rdi
    mov rax, r12
    imul rax, 96                      # Cin * 3 * 8 * sizeof(float)
    mov QWORD PTR [rsp+8], rax
    mov rbx, r9
    shr rbx, 3

.m48_ob_3_t16:
    xor r8d, r8d
.m48_half_3_t16:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi
    xor r10d, r10d
.m48_t_3_t16:
    cmp r10, r13
    jae .m48_next_half_3_t16

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vbroadcastss ymm2, DWORD PTR [rdi+4]
    vmovaps ymm3, ymm2
    vbroadcastss ymm4, DWORD PTR [rdi+8]
    vmovaps ymm5, ymm4
    vbroadcastss ymm6, DWORD PTR [rdi+12]
    vmovaps ymm7, ymm6

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m48_ci_3_t16:
    mov rdi, r11
    M48_K3_T16_STEP
    M48_K3_T16_STEP
    M48_K3_T16_STEP
    add r11, r14
    dec r9
    jnz .m48_ci_3_t16

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2
    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm2
    vmovups YMMWORD PTR [rdi+32], ymm3
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm4
    vmovups YMMWORD PTR [rdi+32], ymm5
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7

    add r10, 16
    jmp .m48_t_3_t16

.m48_next_half_3_t16:
    add r8, 16
    cmp r8, 32
    jb .m48_half_3_t16
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .m48_ob_3_t16

.m48_done_3_t16:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t16_k3,.-ds_conv1d_nct_f32_avx2_oc4_t16_k3

.macro M48_K3_T24_STEP
    vmovups ymm12, YMMWORD PTR [rdi]
    vmovups ymm13, YMMWORD PTR [rdi+32]
    vmovups ymm14, YMMWORD PTR [rdi+64]

    vbroadcastss ymm15, DWORD PTR [rax+0]
    vfmadd231ps ymm0, ymm12, ymm15
    vfmadd231ps ymm1, ymm13, ymm15
    vfmadd231ps ymm2, ymm14, ymm15
    vbroadcastss ymm15, DWORD PTR [rax+4]
    vfmadd231ps ymm3, ymm12, ymm15
    vfmadd231ps ymm4, ymm13, ymm15
    vfmadd231ps ymm5, ymm14, ymm15
    vbroadcastss ymm15, DWORD PTR [rax+8]
    vfmadd231ps ymm6, ymm12, ymm15
    vfmadd231ps ymm7, ymm13, ymm15
    vfmadd231ps ymm8, ymm14, ymm15
    vbroadcastss ymm15, DWORD PTR [rax+12]
    vfmadd231ps ymm9, ymm12, ymm15
    vfmadd231ps ymm10, ymm13, ymm15
    vfmadd231ps ymm11, ymm14, ymm15

    add rdi, r15
    add rax, 32
.endm

.globl ds_conv1d_nct_f32_avx2_oc4_t24_k3
.type ds_conv1d_nct_f32_avx2_oc4_t24_k3,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k3:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 24

    test r8, r8
    jz .m48_done_3_t24
    test r9, r9
    jz .m48_done_3_t24
    test r9, 7
    jnz .m48_done_3_t24

    mov r12, r8
    mov r13, QWORD PTR [rbp+24]
    mov r14, QWORD PTR [rbp+32]
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]
    shl r15, 2
    mov QWORD PTR [rsp+0], rdi
    mov rax, r12
    imul rax, 96
    mov QWORD PTR [rsp+8], rax
    mov rbx, r9
    shr rbx, 3

.m48_ob_3_t24:
    xor r8d, r8d
.m48_half_3_t24:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi
    xor r10d, r10d
.m48_t_3_t24:
    cmp r10, r13
    jae .m48_next_half_3_t24

    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9

    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m48_ci_3_t24:
    mov rdi, r11
    M48_K3_T24_STEP
    M48_K3_T24_STEP
    M48_K3_T24_STEP
    add r11, r14
    dec r9
    jnz .m48_ci_3_t24

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2
    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m48_t_3_t24

.m48_next_half_3_t24:
    add r8, 16
    cmp r8, 32
    jb .m48_half_3_t24
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    dec rbx
    jnz .m48_ob_3_t24

.m48_done_3_t24:
    vzeroupper
    add rsp, 24
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k3,.-ds_conv1d_nct_f32_avx2_oc4_t24_k3


# M49 selective residual-store K=7: oc4 x t24, same arithmetic order as the
# corresponding M45/M46/M48 kernel, with residual added only at the final store.
.globl ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual
.type ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    test r8, r8
    jz .m49r7_done
    test r9, r9
    jz .m49r7_done
    test r9, 7
    jnz .m49r7_done
    mov r12, r8
    mov r13, QWORD PTR [rbp+24]       # Tout
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov QWORD PTR [rsp+0], rdi        # x base
    mov rax, r12
    imul rax, 224
    mov QWORD PTR [rsp+8], rax        # packed bytes / oc8
    mov rax, QWORD PTR [rbp+48]
    mov QWORD PTR [rsp+24], rax       # residual current oc8 base
    mov rbx, r9
    shr rbx, 3
.m49r7_ob:
    xor r8d, r8d
.m49r7_half:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi       # y half base
    xor r10d, r10d
.m49r7_t:
    cmp r10, r13
    jae .m49r7_next_half
    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9
    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m49r7_ci:
    mov rdi, r11
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    M45_K7_STEP
    add r11, r14
    dec r9
    jnz .m49r7_ci

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2
    mov rax, QWORD PTR [rsp+24]
    mov r9, r8
    imul r9, r13
    add rax, r9
    lea rax, [rax+r10*4]

    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm0, ymm0, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm1, ymm1, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm2, ymm2, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm3, ymm3, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm4, ymm4, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm5, ymm5, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm6, ymm6, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm7, ymm7, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm8, ymm8, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm9, ymm9, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm10, ymm10, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm11, ymm11, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m49r7_t
.m49r7_next_half:
    add r8, 16
    cmp r8, 32
    jb .m49r7_half
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    add QWORD PTR [rsp+24], rax
    dec rbx
    jnz .m49r7_ob
.m49r7_done:
    vzeroupper
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual,.-ds_conv1d_nct_f32_avx2_oc4_t24_k7_residual


# M49 selective residual-store K=11: oc4 x t24, same arithmetic order as the
# corresponding M45/M46/M48 kernel, with residual added only at the final store.
.globl ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual
.type ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    test r8, r8
    jz .m49r11_done
    test r9, r9
    jz .m49r11_done
    test r9, 7
    jnz .m49r11_done
    mov r12, r8
    mov r13, QWORD PTR [rbp+24]       # Tout
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov QWORD PTR [rsp+0], rdi        # x base
    mov rax, r12
    imul rax, 352
    mov QWORD PTR [rsp+8], rax        # packed bytes / oc8
    mov rax, QWORD PTR [rbp+48]
    mov QWORD PTR [rsp+24], rax       # residual current oc8 base
    mov rbx, r9
    shr rbx, 3
.m49r11_ob:
    xor r8d, r8d
.m49r11_half:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi       # y half base
    xor r10d, r10d
.m49r11_t:
    cmp r10, r13
    jae .m49r11_next_half
    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9
    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m49r11_ci:
    mov rdi, r11
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    M46_K11_STEP
    add r11, r14
    dec r9
    jnz .m49r11_ci

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2
    mov rax, QWORD PTR [rsp+24]
    mov r9, r8
    imul r9, r13
    add rax, r9
    lea rax, [rax+r10*4]

    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm0, ymm0, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm1, ymm1, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm2, ymm2, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm3, ymm3, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm4, ymm4, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm5, ymm5, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm6, ymm6, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm7, ymm7, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm8, ymm8, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm9, ymm9, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm10, ymm10, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm11, ymm11, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m49r11_t
.m49r11_next_half:
    add r8, 16
    cmp r8, 32
    jb .m49r11_half
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    add QWORD PTR [rsp+24], rax
    dec rbx
    jnz .m49r11_ob
.m49r11_done:
    vzeroupper
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual,.-ds_conv1d_nct_f32_avx2_oc4_t24_k11_residual


# M49 selective residual-store K=3: oc4 x t24, same arithmetic order as the
# corresponding M45/M46/M48 kernel, with residual added only at the final store.
.globl ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual
.type ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual,@function
ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 32
    test r8, r8
    jz .m49r3_done
    test r9, r9
    jz .m49r3_done
    test r9, 7
    jnz .m49r3_done
    mov r12, r8
    mov r13, QWORD PTR [rbp+24]       # Tout
    mov r14, QWORD PTR [rbp+32]       # Tp
    shl r14, 2
    mov r15, QWORD PTR [rbp+40]       # dilation
    shl r15, 2
    mov QWORD PTR [rsp+0], rdi        # x base
    mov rax, r12
    imul rax, 96
    mov QWORD PTR [rsp+8], rax        # packed bytes / oc8
    mov rax, QWORD PTR [rbp+48]
    mov QWORD PTR [rsp+24], rax       # residual current oc8 base
    mov rbx, r9
    shr rbx, 3
.m49r3_ob:
    xor r8d, r8d
.m49r3_half:
    mov rdi, r8
    imul rdi, r13
    add rdi, rcx
    mov QWORD PTR [rsp+16], rdi       # y half base
    xor r10d, r10d
.m49r3_t:
    cmp r10, r13
    jae .m49r3_next_half
    lea rdi, [rdx+r8]
    vbroadcastss ymm0, DWORD PTR [rdi+0]
    vmovaps ymm1, ymm0
    vmovaps ymm2, ymm0
    vbroadcastss ymm3, DWORD PTR [rdi+4]
    vmovaps ymm4, ymm3
    vmovaps ymm5, ymm3
    vbroadcastss ymm6, DWORD PTR [rdi+8]
    vmovaps ymm7, ymm6
    vmovaps ymm8, ymm6
    vbroadcastss ymm9, DWORD PTR [rdi+12]
    vmovaps ymm10, ymm9
    vmovaps ymm11, ymm9
    mov r11, QWORD PTR [rsp+0]
    lea r11, [r11+r10*4]
    mov rax, rsi
    add rax, r8
    mov r9, r12
.m49r3_ci:
    mov rdi, r11
    M48_K3_T24_STEP
    M48_K3_T24_STEP
    M48_K3_T24_STEP
    add r11, r14
    dec r9
    jnz .m49r3_ci

    mov rdi, QWORD PTR [rsp+16]
    lea rdi, [rdi+r10*4]
    mov r11, r13
    shl r11, 2
    mov rax, QWORD PTR [rsp+24]
    mov r9, r8
    imul r9, r13
    add rax, r9
    lea rax, [rax+r10*4]

    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm0, ymm0, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm0
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm1, ymm1, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm1
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm2, ymm2, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm2
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm3, ymm3, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm3
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm4, ymm4, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm4
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm5, ymm5, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm5
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm6, ymm6, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm6
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm7, ymm7, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm7
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm8, ymm8, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm8
    add rdi, r11
    add rax, r11
    vmovups ymm12, YMMWORD PTR [rax+0]
    vaddps ymm9, ymm9, ymm12
    vmovups YMMWORD PTR [rdi+0], ymm9
    vmovups ymm12, YMMWORD PTR [rax+32]
    vaddps ymm10, ymm10, ymm12
    vmovups YMMWORD PTR [rdi+32], ymm10
    vmovups ymm12, YMMWORD PTR [rax+64]
    vaddps ymm11, ymm11, ymm12
    vmovups YMMWORD PTR [rdi+64], ymm11

    add r10, 24
    jmp .m49r3_t
.m49r3_next_half:
    add r8, 16
    cmp r8, 32
    jb .m49r3_half
    add rsi, QWORD PTR [rsp+8]
    add rdx, 32
    mov rax, r13
    shl rax, 5
    add rcx, rax
    add QWORD PTR [rsp+24], rax
    dec rbx
    jnz .m49r3_ob
.m49r3_done:
    vzeroupper
    add rsp, 32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual,.-ds_conv1d_nct_f32_avx2_oc4_t24_k3_residual
