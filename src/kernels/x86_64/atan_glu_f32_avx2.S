.intel_syntax noprefix
.text

# void ds_atan_glu_f32_avx2(
#   const float *x2, // rdi [M,2N], row layout [left N | gate N]
#   float *y,        // rsi [M,N]
#   size_t M,        // rdx
#   size_t N         // rcx
# );
#
# AVX2/FMA implementation of y = left * atan(gate).
# atan is range-reduced to |r| <= tan(pi/8) and evaluated with the odd
# Taylor polynomial through r^13.  On float32 this is accurate to well below
# 1e-5 over ordinary finite inputs while avoiding libm calls in the hot path.
# Fast path requires N % 8 == 0 (all current LYNXNet2 channel counts satisfy it).

.globl ds_atan_glu_f32_avx2
.type ds_atan_glu_f32_avx2,@function
ds_atan_glu_f32_avx2:
    test rdx, rdx
    jz .done
    test rcx, rcx
    jz .done
    test rcx, 7
    jnz .done

    mov r8, rcx
    shl r8, 2                      # N bytes
    mov r9, r8
    shl r9, 1                      # 2N bytes, input row stride

    vbroadcastss ymm15, DWORD PTR [rip + .L_abs_mask]
    vbroadcastss ymm14, DWORD PTR [rip + .L_sign_mask]
    vbroadcastss ymm13, DWORD PTR [rip + .L_tan_pi_8]
    vbroadcastss ymm12, DWORD PTR [rip + .L_tan_3pi_8]
    vbroadcastss ymm11, DWORD PTR [rip + .L_one]

.row:
    mov r10, rdi                   # left
    lea r11, [rdi + r8]            # gate
    mov rax, rsi
    xor r8d, r8d                   # byte offset within N (safe: recompute row stride below)

.vec:
    cmp r8, rcx
    jae .next_row
    # r8 is element index here
    vmovups ymm0, YMMWORD PTR [r11 + r8*4]   # gate
    vmovaps ymm1, ymm0
    vandps ymm1, ymm1, ymm15                  # ax

    # masks: small ax <= tan(pi/8), large ax > tan(3pi/8)
    vcmpps ymm2, ymm1, ymm13, 2               # LE_OS
    vcmpps ymm3, ymm1, ymm12, 14              # GT_OS

    # r_mid = (ax - 1) / (ax + 1)
    vsubps ymm4, ymm1, ymm11
    vaddps ymm5, ymm1, ymm11
    vdivps ymm4, ymm4, ymm5

    # r_large = 1 / ax
    vdivps ymm5, ymm11, ymm1

    # r = mid by default; small -> ax; large -> 1/ax
    vblendvps ymm4, ymm4, ymm1, ymm2
    vblendvps ymm4, ymm4, ymm5, ymm3

    # atan(r) = r * (1 + r^2*(-1/3 + r^2*(1/5 + ... + r^2*(1/13))))
    vmulps ymm5, ymm4, ymm4                    # r2
    vbroadcastss ymm6, DWORD PTR [rip + .L_c13]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c11_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c9_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c7_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c5_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c3_vec]
    vfmadd213ps ymm6, ymm5, YMMWORD PTR [rip + .L_c1_vec]
    vmulps ymm6, ymm6, ymm4                    # p = atan(r)

    # mid = pi/4 + p ; large = pi/2 - p ; small = p
    vaddps ymm7, ymm6, YMMWORD PTR [rip + .L_pi4_vec]
    vmovaps ymm8, YMMWORD PTR [rip + .L_pi2_vec]
    vsubps ymm8, ymm8, ymm6
    vblendvps ymm7, ymm7, ymm6, ymm2           # small
    vblendvps ymm7, ymm7, ymm8, ymm3           # large

    # restore sign of original gate
    vandps ymm0, ymm0, ymm14
    vxorps ymm7, ymm7, ymm0

    vmovups ymm9, YMMWORD PTR [r10 + r8*4]
    vmulps ymm9, ymm9, ymm7
    vmovups YMMWORD PTR [rax + r8*4], ymm9

    add r8, 8
    jmp .vec

.next_row:
    # recompute byte strides from N=rcx
    lea r8, [rcx*4]
    lea r9, [r8+r8]
    add rdi, r9
    add rsi, r8
    dec rdx
    jnz .row

.done:
    vzeroupper
    ret
.size ds_atan_glu_f32_avx2,.-ds_atan_glu_f32_avx2

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
