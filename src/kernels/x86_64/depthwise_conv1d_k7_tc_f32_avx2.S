.intel_syntax noprefix
.text

# void ds_depthwise_conv1d_k7_tc_f32_avx2(
#   const float *x, const float *weight_tap_major, const float *bias,
#   float *y, size_t T, size_t C);
# Native [T,C] depthwise Conv1d(k=7,pad=3). Weight layout [7,C].
# Interior: 4 adjacent time rows x 8 channels, reusing one weight vector.
.globl ds_depthwise_conv1d_k7_tc_f32_avx2
.type ds_depthwise_conv1d_k7_tc_f32_avx2,@function
ds_depthwise_conv1d_k7_tc_f32_avx2:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp,32
    mov r12,r8                    # T
    mov r13,r9                    # C
    test r12,r12
    jz .done
    test r13,r13
    jz .done
    mov r14,r13
    and r14,-8                    # vector channel end
    mov r15,r13
    shl r15,2                     # row/tap stride bytes
    xor r10d,r10d                 # t
    cmp r12,7
    jb .all_generic
.left:
    cmp r10,3
    jae .interior_entry
    call .generic_row
    inc r10
    jmp .left
.interior_entry:
    # four-row block valid while t+3 <= T-4 => t <= T-7
    mov r11,r12
    sub r11,7
    mov QWORD PTR [rsp],r11
.i4_loop:
    cmp r10,QWORD PTR [rsp]
    ja .rest
    xor eax,eax
.i4_cvec:
    cmp rax,r14
    jae .i4_ctail
    vmovups ymm0,YMMWORD PTR [rdx+rax*4]
    vmovaps ymm1,ymm0
    vmovaps ymm2,ymm0
    vmovaps ymm3,ymm0
    mov r8,r10
    sub r8,3
    imul r8,r15
    lea r8,[rdi+r8]
    lea r8,[r8+rax*4]
    lea r9,[r8+r15]
    lea rbp,[r9+r15]
    lea rbx,[rbp+r15]
    lea r11,[rsi+rax*4]
    mov DWORD PTR [rsp+24],7
.i4_taps:
    vmovups ymm4,YMMWORD PTR [r11]
    vmovups ymm5,YMMWORD PTR [r8]
    vfmadd231ps ymm0,ymm5,ymm4
    vmovups ymm5,YMMWORD PTR [r9]
    vfmadd231ps ymm1,ymm5,ymm4
    vmovups ymm5,YMMWORD PTR [rbp]
    vfmadd231ps ymm2,ymm5,ymm4
    vmovups ymm5,YMMWORD PTR [rbx]
    vfmadd231ps ymm3,ymm5,ymm4
    add r8,r15
    add r9,r15
    add rbp,r15
    add rbx,r15
    add r11,r15
    sub DWORD PTR [rsp+24],1
    jnz .i4_taps
    mov r8,r10
    imul r8,r15
    add r8,rcx
    lea r8,[r8+rax*4]
    vmovups YMMWORD PTR [r8],ymm0
    vmovups YMMWORD PTR [r8+r15],ymm1
    lea r9,[r8+r15*2]
    vmovups YMMWORD PTR [r9],ymm2
    add r9,r15
    vmovups YMMWORD PTR [r9],ymm3
    add rax,8
    jmp .i4_cvec
.i4_ctail:
    cmp rax,r13
    jae .i4_next
.i4_scalar:
    vmovss xmm0,DWORD PTR [rdx+rax*4]
    vmovaps xmm1,xmm0
    vmovaps xmm2,xmm0
    vmovaps xmm3,xmm0
    mov r8,r10
    sub r8,3
    imul r8,r15
    lea r8,[rdi+r8]
    lea r8,[r8+rax*4]
    lea r9,[r8+r15]
    lea rbp,[r9+r15]
    lea rbx,[rbp+r15]
    lea r11,[rsi+rax*4]
    mov DWORD PTR [rsp+24],7
.i4_scalar_taps:
    vmovss xmm4,DWORD PTR [r11]
    vfmadd231ss xmm0,xmm4,DWORD PTR [r8]
    vfmadd231ss xmm1,xmm4,DWORD PTR [r9]
    vfmadd231ss xmm2,xmm4,DWORD PTR [rbp]
    vfmadd231ss xmm3,xmm4,DWORD PTR [rbx]
    add r8,r15
    add r9,r15
    add rbp,r15
    add rbx,r15
    add r11,r15
    sub DWORD PTR [rsp+24],1
    jnz .i4_scalar_taps
    mov r8,r10
    imul r8,r15
    add r8,rcx
    lea r8,[r8+rax*4]
    vmovss DWORD PTR [r8],xmm0
    vmovss DWORD PTR [r8+r15],xmm1
    lea r9,[r8+r15*2]
    vmovss DWORD PTR [r9],xmm2
    add r9,r15
    vmovss DWORD PTR [r9],xmm3
    inc rax
    cmp rax,r13
    jb .i4_scalar
.i4_next:
    add r10,4
    jmp .i4_loop
.rest:
    cmp r10,r12
    jae .done
.rest_loop:
    call .generic_row
    inc r10
    cmp r10,r12
    jb .rest_loop
    jmp .done
.all_generic:
    xor r10d,r10d
.ag_loop:
    call .generic_row
    inc r10
    cmp r10,r12
    jb .ag_loop
    jmp .done

# generic row for pad=3,k=7
.generic_row:
    # start=max(0,3-t)
    xor eax,eax
    cmp r10,3
    jae .gr_start
    mov rax,3
    sub rax,r10
.gr_start:
    # end=min(7,T+3-t)
    mov r11,r12
    add r11,3
    sub r11,r10
    cmp r11,7
    jbe .gr_end
    mov r11,7
.gr_end:
    mov rbp,r11
    sub rbp,rax
    mov QWORD PTR [rsp+24],rbp
    # x row=t+start-3
    mov r11,r10
    add r11,rax
    sub r11,3
    imul r11,r15
    lea r11,[rdi+r11]
    mov QWORD PTR [rsp+8],r11
    imul rax,r15
    lea r11,[rsi+rax]
    mov QWORD PTR [rsp+16],r11
    mov rbx,r10
    imul rbx,r15
    add rbx,rcx
    xor eax,eax
.gr_vec:
    cmp rax,r14
    jae .gr_tail
    vmovups ymm0,YMMWORD PTR [rdx+rax*4]
    mov r8,QWORD PTR [rsp+8]
    mov r9,QWORD PTR [rsp+16]
    lea r8,[r8+rax*4]
    lea r9,[r9+rax*4]
    mov r11,QWORD PTR [rsp+24]
.gr_taps:
    vmovups ymm1,YMMWORD PTR [r8]
    vfmadd231ps ymm0,ymm1,YMMWORD PTR [r9]
    add r8,r15
    add r9,r15
    dec r11
    jnz .gr_taps
    vmovups YMMWORD PTR [rbx+rax*4],ymm0
    add rax,8
    jmp .gr_vec
.gr_tail:
    cmp rax,r13
    jae .gr_ret
.gr_scalar:
    vmovss xmm0,DWORD PTR [rdx+rax*4]
    mov r8,QWORD PTR [rsp+8]
    mov r9,QWORD PTR [rsp+16]
    lea r8,[r8+rax*4]
    lea r9,[r9+rax*4]
    mov r11,QWORD PTR [rsp+24]
.gr_scalar_taps:
    vmovss xmm1,DWORD PTR [r8]
    vfmadd231ss xmm0,xmm1,DWORD PTR [r9]
    add r8,r15
    add r9,r15
    dec r11
    jnz .gr_scalar_taps
    vmovss DWORD PTR [rbx+rax*4],xmm0
    inc rax
    cmp rax,r13
    jb .gr_scalar
.gr_ret:
    ret
.done:
    vzeroupper
    add rsp,32
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_depthwise_conv1d_k7_tc_f32_avx2,.-ds_depthwise_conv1d_k7_tc_f32_avx2
.section .note.GNU-stack,"",@progbits
