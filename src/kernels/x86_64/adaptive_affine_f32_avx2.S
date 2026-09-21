.intel_syntax noprefix
.text

# void ds_adaptive_affine_f32_avx2(float *normalized_tc,
#                                  const float *shift_scale_tc,
#                                  size_t rows, size_t channels)
# rdi=normalized rsi=shift_scale rdx=rows rcx=channels
.globl ds_adaptive_affine_f32_avx2
.type ds_adaptive_affine_f32_avx2,@function
ds_adaptive_affine_f32_avx2:
    push r12
    test rdx, rdx
    jz .Ldone
    test rcx, rcx
    jz .Ldone
    xor r8d, r8d
.Lrow:
    mov r9, r8
    imul r9, rcx
    lea r10, [rdi+r9*4]
    shl r9, 1
    lea r11, [rsi+r9*4]
    lea rax, [r11+rcx*4]
    xor r9d, r9d
    mov r12, rcx
    and r12, -8
.Lvec:
    cmp r9, r12
    jae .Ltail
    vmovups ymm0, YMMWORD PTR [r10+r9*4]
    vmovups ymm1, YMMWORD PTR [rax+r9*4]
    vfmadd213ps ymm0, ymm1, YMMWORD PTR [r11+r9*4]
    vmovups YMMWORD PTR [r10+r9*4], ymm0
    add r9, 8
    jmp .Lvec
.Ltail:
    cmp r9, rcx
    jae .Lnext
    vmovss xmm0, DWORD PTR [r10+r9*4]
    vmovss xmm1, DWORD PTR [rax+r9*4]
    vfmadd213ss xmm0, xmm1, DWORD PTR [r11+r9*4]
    vmovss DWORD PTR [r10+r9*4], xmm0
    inc r9
    jmp .Ltail
.Lnext:
    inc r8
    cmp r8, rdx
    jb .Lrow
.Ldone:
    vzeroupper
    pop r12
    ret
.size ds_adaptive_affine_f32_avx2,.-ds_adaptive_affine_f32_avx2
.section .note.GNU-stack,"",@progbits
