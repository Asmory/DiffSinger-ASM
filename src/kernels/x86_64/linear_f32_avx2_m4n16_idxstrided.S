.intel_syntax noprefix
.text

# void ds_linear_f32_avx2_m4n16_idxstrided(
#   const float *x,          // rdi [M,K]
#   const float *w_packed,   // rsi packed16: per block/per k W0[8],W1[8]
#   const float *bias,       // rdx [N]
#   float *y,                // rcx [M,N]
#   size_t M,                // r8
#   size_t N,                // r9
#   size_t K,                // [rbp+16]
#   size_t y_stride          // [rbp+24], elements
# );
# N % 16 == 0, K arbitrary. 4x16 micro-kernel.

.globl ds_linear_f32_avx2_m4n16_idxstrided
.type ds_linear_f32_avx2_m4n16_idxstrided,@function
ds_linear_f32_avx2_m4n16_idxstrided:
 push rbp
 mov rbp,rsp
 push rbx
 push r12
 push r13
 push r14
 push r15
 sub rsp,16
 mov r13,QWORD PTR [rbp+16]            # K
 test r8,r8
 jz .done
 test r9,r9
 jz .done
 test r9,15
 jnz .done
 test r13,r13
 jz .done
 mov r12,r9
 shr r12,4                             # nblocks
 mov r14,r13
 shl r14,2                             # x stride
 mov r15,QWORD PTR [rbp+24]
 shl r15,2                             # full y row stride
 mov QWORD PTR [rsp+8],rsi
.main4:
 cmp r8,4
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
 xor r13d,r13d                         # one shared byte offset for all 4 rows
.k4:
 vmovups ymm9,YMMWORD PTR [rsi]
 vmovups ymm10,YMMWORD PTR [rsi+32]
 vbroadcastss ymm8,DWORD PTR [r10+r13]
 vfmadd231ps ymm0,ymm8,ymm9
 vfmadd231ps ymm1,ymm8,ymm10
 vbroadcastss ymm8,DWORD PTR [r11+r13]
 vfmadd231ps ymm2,ymm8,ymm9
 vfmadd231ps ymm3,ymm8,ymm10
 vbroadcastss ymm8,DWORD PTR [rax+r13]
 vfmadd231ps ymm4,ymm8,ymm9
 vfmadd231ps ymm5,ymm8,ymm10
 vbroadcastss ymm8,DWORD PTR [rbx+r13]
 vfmadd231ps ymm6,ymm8,ymm9
 vfmadd231ps ymm7,ymm8,ymm10
 add rsi,64
 add r13,4
 cmp r13,r14
 jb .k4
 vmovups YMMWORD PTR [rcx],ymm0
 vmovups YMMWORD PTR [rcx+32],ymm1
 lea r10,[rcx+r15]
 vmovups YMMWORD PTR [r10],ymm2
 vmovups YMMWORD PTR [r10+32],ymm3
 lea r11,[r15+r15]
 lea r10,[rcx+r11]
 vmovups YMMWORD PTR [r10],ymm4
 vmovups YMMWORD PTR [r10+32],ymm5
 lea r11,[r15+r15*2]
 lea r10,[rcx+r11]
 vmovups YMMWORD PTR [r10],ymm6
 vmovups YMMWORD PTR [r10+32],ymm7
 add rdx,64
 add rcx,64
 dec QWORD PTR [rsp]
 jnz .nb4
 mov rsi,QWORD PTR [rsp+8]
 mov r10,r9
 shl r10,2                             # tile bytes
 sub rdx,r10
 sub rcx,r10                           # rewind from end of tile to row0 tile origin
 lea r11,[r15+r15*2]
 add rcx,r11
 add rcx,r15                           # +4 full rows
 lea r10,[r14+r14*2]
 add rdi,r10
 add rdi,r14
 sub r8,4
 jmp .main4
.tailcheck:
 test r8,r8
 jz .done
.tailrow:
 mov QWORD PTR [rsp],r12
.nb1:
 vmovups ymm0,YMMWORD PTR [rdx]
 vmovups ymm1,YMMWORD PTR [rdx+32]
 mov r10,rdi
 xor r13d,r13d
.k1:
 vmovups ymm9,YMMWORD PTR [rsi]
 vmovups ymm10,YMMWORD PTR [rsi+32]
 vbroadcastss ymm8,DWORD PTR [r10+r13]
 vfmadd231ps ymm0,ymm8,ymm9
 vfmadd231ps ymm1,ymm8,ymm10
 add rsi,64
 add r13,4
 cmp r13,r14
 jb .k1
 vmovups YMMWORD PTR [rcx],ymm0
 vmovups YMMWORD PTR [rcx+32],ymm1
 add rdx,64
 add rcx,64
 dec QWORD PTR [rsp]
 jnz .nb1
 mov rsi,QWORD PTR [rsp+8]
 mov r10,r9
 shl r10,2
 sub rdx,r10
 sub rcx,r10                           # rewind tile width
 add rcx,r15                           # next full row, same N tile
 add rdi,r14
 dec r8
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
.size ds_linear_f32_avx2_m4n16_idxstrided,.-ds_linear_f32_avx2_m4n16_idxstrided
.section .note.GNU-stack,"",@progbits
