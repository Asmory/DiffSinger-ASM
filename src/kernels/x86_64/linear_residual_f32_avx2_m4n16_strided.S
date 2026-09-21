.intel_syntax noprefix
.text
# void ds_linear_residual_f32_avx2_m4n16_strided(
#   const float *x, const float *w_packed, const float *bias,
#   const float *residual, float *y, size_t M, size_t N_tile,
#   size_t K, size_t row_stride);
# N%16==0. Packing per N=16 block/per k: W0[8], W1[8].
.globl ds_linear_residual_f32_avx2_m4n16_strided
.type ds_linear_residual_f32_avx2_m4n16_strided,@function
ds_linear_residual_f32_avx2_m4n16_strided:
 push rbp
 mov rbp,rsp
 push rbx
 push r12
 push r13
 push r14
 push r15
 sub rsp,16
 mov rax,QWORD PTR [rbp+16]
 mov r13,QWORD PTR [rbp+24]
 test r9,r9
 jz .done
 test rax,15
 jnz .done
 test r13,r13
 jz .done
 mov r12,rax
 shr r12,4
 mov r14,r13
 shl r14,2
 mov r15,QWORD PTR [rbp+32]
 shl r15,2                            # full residual/output row stride
 mov QWORD PTR [rsp+8],rsi
.main4:
 cmp r9,4
 jb .tailcheck
 mov QWORD PTR [rsp],r12
.nb4:
 vmovups ymm0,YMMWORD PTR [rdx]
 vmovups ymm1,YMMWORD PTR [rdx+32]
 vmovaps ymm2,ymm0
 vmovaps ymm3,ymm1
 vmovaps ymm4,ymm0
 vmovaps ymm5,ymm1
 vmovaps ymm6,ymm0
 vmovaps ymm7,ymm1
 mov r10,rdi
 lea r11,[rdi+r14]
 lea rax,[r11+r14]
 lea rbx,[rax+r14]
 mov r13,QWORD PTR [rbp+24]
.k4:
 vmovups ymm8,YMMWORD PTR [rsi]
 vmovups ymm9,YMMWORD PTR [rsi+32]
 vbroadcastss ymm10,DWORD PTR [r10]
 vfmadd231ps ymm0,ymm10,ymm8
 vfmadd231ps ymm1,ymm10,ymm9
 vbroadcastss ymm10,DWORD PTR [r11]
 vfmadd231ps ymm2,ymm10,ymm8
 vfmadd231ps ymm3,ymm10,ymm9
 vbroadcastss ymm10,DWORD PTR [rax]
 vfmadd231ps ymm4,ymm10,ymm8
 vfmadd231ps ymm5,ymm10,ymm9
 vbroadcastss ymm10,DWORD PTR [rbx]
 vfmadd231ps ymm6,ymm10,ymm8
 vfmadd231ps ymm7,ymm10,ymm9
 add r10,4
 add r11,4
 add rax,4
 add rbx,4
 add rsi,64
 dec r13
 jnz .k4
 # row0
 vaddps ymm0,ymm0,YMMWORD PTR [rcx]
 vaddps ymm1,ymm1,YMMWORD PTR [rcx+32]
 vmovups YMMWORD PTR [r8],ymm0
 vmovups YMMWORD PTR [r8+32],ymm1
 # row1
 lea r10,[rcx+r15]
 lea r11,[r8+r15]
 vaddps ymm2,ymm2,YMMWORD PTR [r10]
 vaddps ymm3,ymm3,YMMWORD PTR [r10+32]
 vmovups YMMWORD PTR [r11],ymm2
 vmovups YMMWORD PTR [r11+32],ymm3
 # row2
 lea r10,[r15+r15]
 lea r11,[rcx+r10]
 lea rax,[r8+r10]
 vaddps ymm4,ymm4,YMMWORD PTR [r11]
 vaddps ymm5,ymm5,YMMWORD PTR [r11+32]
 vmovups YMMWORD PTR [rax],ymm4
 vmovups YMMWORD PTR [rax+32],ymm5
 # row3
 lea r10,[r15+r15*2]
 lea r11,[rcx+r10]
 lea rax,[r8+r10]
 vaddps ymm6,ymm6,YMMWORD PTR [r11]
 vaddps ymm7,ymm7,YMMWORD PTR [r11+32]
 vmovups YMMWORD PTR [rax],ymm6
 vmovups YMMWORD PTR [rax+32],ymm7
 add rdx,64
 add rcx,64
 add r8,64
 dec QWORD PTR [rsp]
 jnz .nb4
 mov rsi,QWORD PTR [rsp+8]
 mov r10,QWORD PTR [rbp+16]
 shl r10,2                            # tile bytes
 sub rdx,r10
 sub rcx,r10
 sub r8,r10                           # rewind N tile
 lea r11,[r15+r15*2]
 add rcx,r11
 add rcx,r15                          # +4 full rows
 add r8,r11
 add r8,r15
 lea r10,[r14+r14*2]
 add rdi,r10
 add rdi,r14
 sub r9,4
 jmp .main4
.tailcheck:
 test r9,r9
 jz .done
.tailrow:
 mov QWORD PTR [rsp],r12
.nb1:
 vmovups ymm0,YMMWORD PTR [rdx]
 vmovups ymm1,YMMWORD PTR [rdx+32]
 mov r10,rdi
 mov r13,QWORD PTR [rbp+24]
.k1:
 vmovups ymm8,YMMWORD PTR [rsi]
 vmovups ymm9,YMMWORD PTR [rsi+32]
 vbroadcastss ymm10,DWORD PTR [r10]
 vfmadd231ps ymm0,ymm10,ymm8
 vfmadd231ps ymm1,ymm10,ymm9
 add r10,4
 add rsi,64
 dec r13
 jnz .k1
 vaddps ymm0,ymm0,YMMWORD PTR [rcx]
 vaddps ymm1,ymm1,YMMWORD PTR [rcx+32]
 vmovups YMMWORD PTR [r8],ymm0
 vmovups YMMWORD PTR [r8+32],ymm1
 add rdx,64
 add rcx,64
 add r8,64
 dec QWORD PTR [rsp]
 jnz .nb1
 mov rsi,QWORD PTR [rsp+8]
 mov r10,QWORD PTR [rbp+16]
 shl r10,2
 sub rdx,r10
 sub rcx,r10
 sub r8,r10                           # rewind N tile
 add rcx,r15
 add r8,r15                           # next full row
 add rdi,r14
 dec r9
 jnz .tailrow
.done:
 vzeroupper
 add rsp,16
 pop r15
 pop r14
 pop r13
 pop r12
 pop rbx
 pop rbp
 ret
.size ds_linear_residual_f32_avx2_m4n16_strided,.-ds_linear_residual_f32_avx2_m4n16_strided
.section .note.GNU-stack,"",@progbits
