.intel_syntax noprefix
.text

# void ds_phoneme_mean_f32_avx2(const float *frame_tc,
#                               const int32_t *durations, float *phoneme_pc,
#                               size_t phonemes, size_t channels)
# rdi=frame rsi=durations rdx=phoneme rcx=P r8=C
.globl ds_phoneme_mean_f32_avx2
.type ds_phoneme_mean_f32_avx2,@function
ds_phoneme_mean_f32_avx2:
    push r12
    push r13
    push r14
    push r15
    test rcx, rcx
    jz .Ldone
    test r8, r8
    jz .Ldone
    xor r9d, r9d
    mov r10, rdi
    mov r11, rdx
.Lphoneme:
    xor r12d, r12d
    vxorps ymm0, ymm0, ymm0
.Lzero_vec:
    lea rax, [r12+8]
    cmp rax, r8
    ja .Lzero_tail
    vmovups YMMWORD PTR [r11+r12*4], ymm0
    mov r12, rax
    jmp .Lzero_vec
.Lzero_tail:
    cmp r12, r8
    jae .Lsum_prep
    mov DWORD PTR [r11+r12*4], 0
    inc r12
    jmp .Lzero_tail
.Lsum_prep:
    mov r15d, DWORD PTR [rsi+r9*4]
    test r15, r15
    jz .Lnext
    xor r13d, r13d
.Lsum_row:
    xor r12d, r12d
.Ladd_vec:
    lea rax, [r12+8]
    cmp rax, r8
    ja .Ladd_tail
    vmovups ymm0, YMMWORD PTR [r11+r12*4]
    vaddps ymm0, ymm0, YMMWORD PTR [r10+r12*4]
    vmovups YMMWORD PTR [r11+r12*4], ymm0
    mov r12, rax
    jmp .Ladd_vec
.Ladd_tail:
    cmp r12, r8
    jae .Lrow_done
    vmovss xmm0, DWORD PTR [r11+r12*4]
    vaddss xmm0, xmm0, DWORD PTR [r10+r12*4]
    vmovss DWORD PTR [r11+r12*4], xmm0
    inc r12
    jmp .Ladd_tail
.Lrow_done:
    lea r10, [r10+r8*4]
    inc r13
    cmp r13, r15
    jb .Lsum_row
    vxorps xmm0, xmm0, xmm0
    vcvtsi2ss xmm0, xmm0, r15
    vbroadcastss ymm1, xmm0
    xor r12d, r12d
.Ldiv_vec:
    lea rax, [r12+8]
    cmp rax, r8
    ja .Ldiv_tail
    vmovups ymm2, YMMWORD PTR [r11+r12*4]
    vdivps ymm2, ymm2, ymm1
    vmovups YMMWORD PTR [r11+r12*4], ymm2
    mov r12, rax
    jmp .Ldiv_vec
.Ldiv_tail:
    cmp r12, r8
    jae .Lnext
    vmovss xmm2, DWORD PTR [r11+r12*4]
    vdivss xmm2, xmm2, xmm0
    vmovss DWORD PTR [r11+r12*4], xmm2
    inc r12
    jmp .Ldiv_tail
.Lnext:
    lea r11, [r11+r8*4]
    inc r9
    cmp r9, rcx
    jb .Lphoneme
.Ldone:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    ret
.size ds_phoneme_mean_f32_avx2,.-ds_phoneme_mean_f32_avx2
.section .note.GNU-stack,"",@progbits
