.intel_syntax noprefix
.text

# void ds_silu_glu_f32_avx2(const float *x2, float *y, size_t M, size_t N)
# x2 rows are [left N | gate N]. Computes left * gate * sigmoid(gate).
.globl ds_silu_glu_f32_avx2
.type ds_silu_glu_f32_avx2,@function
ds_silu_glu_f32_avx2:
    test rdx, rdx
    jz .Lempty
    test rcx, rcx
    jz .Lempty
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    sub rsp, 8
    mov rbx, rdx
    mov rbp, rcx
    mov r12, rdi
    lea r13, [rdi+rcx*4]
    mov r14, rsi

.Lrow:
    xor r15d, r15d
.Lelement:
    vxorps xmm1, xmm1, xmm1
    vsubss xmm0, xmm1, DWORD PTR [r13+r15*4]
    call expf@PLT
    vaddss xmm0, xmm0, DWORD PTR [rip+.Lone]
    vmovss xmm1, DWORD PTR [r13+r15*4]
    vdivss xmm1, xmm1, xmm0
    vmulss xmm1, xmm1, DWORD PTR [r12+r15*4]
    vmovss DWORD PTR [r14+r15*4], xmm1
    inc r15
    cmp r15, rbp
    jb .Lelement
    lea rax, [rbp*8]
    add r12, rax
    add r13, rax
    lea rax, [rbp*4]
    add r14, rax
    dec rbx
    jnz .Lrow

    add rsp, 8
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
.Lempty:
    vzeroupper
    ret
.size ds_silu_glu_f32_avx2,.-ds_silu_glu_f32_avx2

.section .rodata
.align 4
.Lone: .float 1.0
.section .note.GNU-stack,"",@progbits
