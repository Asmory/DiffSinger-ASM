.intel_syntax noprefix
.text

# void ds_rope_f32_avx2(float *x, const float *cosv, const float *sinv,
#                       size_t head_dim, uint32_t interleaved)
# rdi=x rsi=cosv rdx=sinv rcx=head_dim r8d=interleaved
.globl ds_rope_f32_avx2
.type ds_rope_f32_avx2,@function
ds_rope_f32_avx2:
    mov r9, rcx
    shr r9, 1
    test r8d, r8d
    jnz .Linterleaved

    mov r11, r9
    and r11, -8
    xor r10d, r10d
.Lsplit_vec:
    cmp r10, r11
    jae .Lsplit_tail
    vmovups ymm0, YMMWORD PTR [rdi+r10*4]
    lea rax, [r10+r9]
    vmovups ymm1, YMMWORD PTR [rdi+rax*4]
    vmovups ymm2, YMMWORD PTR [rsi+r10*4]
    vmovups ymm3, YMMWORD PTR [rdx+r10*4]
    vmulps ymm4, ymm0, ymm2
    vmulps ymm5, ymm1, ymm3
    vsubps ymm4, ymm4, ymm5
    vmulps ymm6, ymm1, ymm2
    vmulps ymm7, ymm0, ymm3
    vaddps ymm6, ymm6, ymm7
    vmovups YMMWORD PTR [rdi+r10*4], ymm4
    vmovups YMMWORD PTR [rdi+rax*4], ymm6
    add r10, 8
    jmp .Lsplit_vec
.Lsplit_tail:
    cmp r10, r9
    jae .Ldone
    lea rax, [r10+r9]
    vmovss xmm0, DWORD PTR [rdi+r10*4]
    vmovss xmm1, DWORD PTR [rdi+rax*4]
    vmovss xmm2, DWORD PTR [rsi+r10*4]
    vmovss xmm3, DWORD PTR [rdx+r10*4]
    vmulss xmm4, xmm0, xmm2
    vmulss xmm5, xmm1, xmm3
    vsubss xmm4, xmm4, xmm5
    vmulss xmm6, xmm1, xmm2
    vmulss xmm7, xmm0, xmm3
    vaddss xmm6, xmm6, xmm7
    vmovss DWORD PTR [rdi+r10*4], xmm4
    vmovss DWORD PTR [rdi+rax*4], xmm6
    inc r10
    jmp .Lsplit_tail

.Linterleaved:
    xor r10d, r10d
.Linterleaved_loop:
    cmp r10, r9
    jae .Ldone
    lea rax, [r10+r10]
    vmovss xmm0, DWORD PTR [rdi+rax*4]
    vmovss xmm1, DWORD PTR [rdi+rax*4+4]
    vmovss xmm2, DWORD PTR [rsi+r10*4]
    vmovss xmm3, DWORD PTR [rdx+r10*4]
    vmulss xmm4, xmm0, xmm2
    vmulss xmm5, xmm1, xmm3
    vsubss xmm4, xmm4, xmm5
    vmulss xmm6, xmm1, xmm2
    vmulss xmm7, xmm0, xmm3
    vaddss xmm6, xmm6, xmm7
    vmovss DWORD PTR [rdi+rax*4], xmm4
    vmovss DWORD PTR [rdi+rax*4+4], xmm6
    inc r10
    jmp .Linterleaved_loop

.Ldone:
    vzeroupper
    ret
.size ds_rope_f32_avx2,.-ds_rope_f32_avx2
.section .note.GNU-stack,"",@progbits
