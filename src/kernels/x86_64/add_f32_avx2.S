.intel_syntax noprefix
.text
# void ds_add_f32_avx2(const float *a, const float *b, float *y, size_t n)
.globl ds_add_f32_avx2
.type ds_add_f32_avx2,@function
ds_add_f32_avx2:
    test rcx, rcx
    jz .done
    mov r8, rcx
    and r8, -8
    xor r9d, r9d
.vec:
    cmp r9, r8
    jae .tail
    vmovups ymm0, YMMWORD PTR [rdi+r9*4]
    vaddps ymm0, ymm0, YMMWORD PTR [rsi+r9*4]
    vmovups YMMWORD PTR [rdx+r9*4], ymm0
    add r9, 8
    jmp .vec
.tail:
    cmp r9, rcx
    jae .done
    vmovss xmm0, DWORD PTR [rdi+r9*4]
    vaddss xmm0, xmm0, DWORD PTR [rsi+r9*4]
    vmovss DWORD PTR [rdx+r9*4], xmm0
    inc r9
    jmp .tail
.done:
    vzeroupper
    ret
.size ds_add_f32_avx2,.-ds_add_f32_avx2
.section .note.GNU-stack,"",@progbits
