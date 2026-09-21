.intel_syntax noprefix
.text
# void ds_leaky_copy_nct_f32_avx2(const float *x, float *dst,
#                                  size_t C, size_t T, size_t Tp, size_t pad,
#                                  float alpha);
# x is [C,T], dst is [C,Tp]. M39 clears both pad halos in this same pass.
.globl ds_leaky_copy_nct_f32_avx2
.type ds_leaky_copy_nct_f32_avx2,@function
ds_leaky_copy_nct_f32_avx2:
    push r12
    push r13
    push r14
    push r15
    mov r12, rdx                      # C
    mov r13, rcx                      # T
    mov r14, r8                       # Tp
    mov r15, r9                       # pad
    shl r14, 2                        # Tp bytes
    shl r15, 2                        # pad bytes
    test r12, r12
    jz .done
    test r13, r13
    jz .done
    vxorps ymm3, ymm3, ymm3
    vbroadcastss ymm4, xmm0
.c_loop:
    # M39: clear only left/right halos here instead of memset(C*Tp) first.
    mov rax, r15
    shr rax, 2                        # pad floats
    xor r11d, r11d
.zleft_vec:
    mov r9, rax
    sub r9, r11
    cmp r9, 8
    jb .zleft_tail
    vmovups YMMWORD PTR [rsi+r11*4], ymm3
    add r11, 8
    jmp .zleft_vec
.zleft_tail:
    cmp r11, rax
    jae .zright_prep
    vmovss DWORD PTR [rsi+r11*4], xmm3
    inc r11
    jmp .zleft_tail
.zright_prep:
    lea r8, [rsi+r15]
    lea r8, [r8+r13*4]
    xor r11d, r11d
.zright_vec:
    mov r9, rax
    sub r9, r11
    cmp r9, 8
    jb .zright_tail
    vmovups YMMWORD PTR [r8+r11*4], ymm3
    add r11, 8
    jmp .zright_vec
.zright_tail:
    cmp r11, rax
    jae .copy_center
    vmovss DWORD PTR [r8+r11*4], xmm3
    inc r11
    jmp .zright_tail
.copy_center:
    lea r10, [rsi+r15]
    xor r11d, r11d
.vloop:
    mov rax, r13
    sub rax, r11
    cmp rax, 8
    jb .tail
    vmovups ymm0, YMMWORD PTR [rdi+r11*4]
    vmaxps ymm1, ymm0, ymm3
    vminps ymm2, ymm0, ymm3
    vmulps ymm2, ymm2, ymm4
    vaddps ymm1, ymm1, ymm2
    vmovups YMMWORD PTR [r10+r11*4], ymm1
    add r11, 8
    jmp .vloop
.tail:
    cmp r11, r13
    jae .next_c
.sloop:
    vmovss xmm0, DWORD PTR [rdi+r11*4]
    vmaxss xmm1, xmm0, xmm3
    vminss xmm2, xmm0, xmm3
    vmulss xmm2, xmm2, xmm4
    vaddss xmm1, xmm1, xmm2
    vmovss DWORD PTR [r10+r11*4], xmm1
    inc r11
    cmp r11, r13
    jb .sloop
.next_c:
    lea rdi, [rdi+r13*4]
    add rsi, r14
    dec r12
    jnz .c_loop
.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    ret
.size ds_leaky_copy_nct_f32_avx2,.-ds_leaky_copy_nct_f32_avx2
.section .note.GNU-stack,"",@progbits
