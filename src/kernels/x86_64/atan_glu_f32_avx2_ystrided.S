.intel_syntax noprefix
.text

# void ds_atan_glu_f32_avx2_ystrided(
#   const float *x2,       // rdi [M,2N], row layout [left N | gate N]
#   float *y,              // rsi [M,y_stride]
#   size_t M,              // rdx
#   size_t N,              // rcx
#   size_t y_stride        // r8, elements
# );
#
# M9 tile-local ATanGLU. Input is a compact worker-private tile while the
# destination may be a slice of a wider [T,C] tensor. N must be divisible by 8.

.globl ds_atan_glu_f32_avx2_ystrided
.type ds_atan_glu_f32_avx2_ystrided,@function
ds_atan_glu_f32_avx2_ystrided:
    push r12
    push r13
    test rdx, rdx
    jz .done
    test rcx, rcx
    jz .done
    test rcx, 7
    jnz .done

    mov r9, rcx
    shl r9, 2                      # N bytes
    shl r8, 2                      # output row stride bytes
    lea r10, [r9+r9]               # input row stride = 2N bytes

    vbroadcastss ymm15, DWORD PTR [rip + .L_abs_mask]
    vbroadcastss ymm14, DWORD PTR [rip + .L_sign_mask]
    vbroadcastss ymm13, DWORD PTR [rip + .L_tan_pi_8]
    vbroadcastss ymm12, DWORD PTR [rip + .L_tan_3pi_8]
    vbroadcastss ymm11, DWORD PTR [rip + .L_one]

.row:
    mov r11, rdi                    # left ptr
    lea rax, [rdi+r9]               # gate ptr
    mov r12, rsi                    # output ptr
    mov r13, rcx                    # elements remaining

.vec:
    vmovups ymm0, YMMWORD PTR [rax]
    vmovaps ymm1, ymm0
    vandps ymm1, ymm1, ymm15

    vcmpps ymm2, ymm1, ymm13, 2
    vcmpps ymm3, ymm1, ymm12, 14

    vsubps ymm4, ymm1, ymm11
    vaddps ymm5, ymm1, ymm11
    vdivps ymm4, ymm4, ymm5
    vdivps ymm5, ymm11, ymm1
    vblendvps ymm4, ymm4, ymm1, ymm2
    vblendvps ymm4, ymm4, ymm5, ymm3

    vmulps ymm5, ymm4, ymm4
    vbroadcastss ymm6, DWORD PTR [rip + .L_c13]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c11_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c9_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c7_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c5_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c3_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c1_vec]
    vmulps ymm6, ymm6, ymm4

    vaddps ymm7, ymm6, YMMWORD PTR [rip + .L_pi4_vec]
    vmovaps ymm8, YMMWORD PTR [rip + .L_pi2_vec]
    vsubps ymm8, ymm8, ymm6
    vblendvps ymm7, ymm7, ymm6, ymm2
    vblendvps ymm7, ymm7, ymm8, ymm3

    vandps ymm0, ymm0, ymm14
    vxorps ymm7, ymm7, ymm0

    vmovups ymm9, YMMWORD PTR [r11]
    vmulps ymm9, ymm9, ymm7
    vmovups YMMWORD PTR [r12], ymm9

    add r11, 32
    add rax, 32
    add r12, 32
    sub r13, 8
    jnz .vec

    add rdi, r10
    add rsi, r8
    dec rdx
    jnz .row

.done:
    vzeroupper
    pop r13
    pop r12
    ret
.size ds_atan_glu_f32_avx2_ystrided,.-ds_atan_glu_f32_avx2_ystrided

.section .rodata
.align 32
.L_abs_mask: .long 0x7fffffff
.L_sign_mask: .long 0x80000000
.L_tan_pi_8: .float 0.4142135623730950
.L_tan_3pi_8: .float 2.4142135623730950
.L_one: .float 1.0
.L_c13: .float 0.0769230769230769
.align 32
.L_c11_vec: .rept 8; .float -0.0909090909090909; .endr
.L_c9_vec:  .rept 8; .float 0.1111111111111111; .endr
.L_c7_vec:  .rept 8; .float -0.1428571428571429; .endr
.L_c5_vec:  .rept 8; .float 0.2; .endr
.L_c3_vec:  .rept 8; .float -0.3333333333333333; .endr
.L_c1_vec:  .rept 8; .float 1.0; .endr
.L_pi4_vec: .rept 8; .float 0.7853981633974483; .endr
.L_pi2_vec: .rept 8; .float 1.5707963267948966; .endr
.section .note.GNU-stack,"",@progbits
