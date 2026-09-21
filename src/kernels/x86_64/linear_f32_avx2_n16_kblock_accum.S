.intel_syntax noprefix
.text

# void ds_linear_f32_avx2_n16_kblock_accum(
#   const float *x,          // rdi: points at x[:, k0], physical row stride = K_stride
#   const float *w_chunk,    // rsi: [K_len,16], packed16 one output block
#   const float *bias,       // rdx: [16], used only when init!=0
#   float *y,                // rcx: points at output N-block
#   size_t M,                // r8
#   size_t K_stride,         // r9, full input row stride in elements
#   size_t K_len,            // [rbp+16]
#   size_t y_stride,         // [rbp+24], full output row stride in elements
#   int init                 // [rbp+32], nonzero=bias init, zero=accumulate existing y
# );
#
# M15 cache-blocking primitive. It computes exactly one 16-output packed block
# for a contiguous K chunk. Calling chunks in increasing K order preserves the
# original FP32 FMA accumulation order; stores/reloads between chunks are exact.

.globl ds_linear_f32_avx2_n16_kblock_accum
.type ds_linear_f32_avx2_n16_kblock_accum,@function
ds_linear_f32_avx2_n16_kblock_accum:
    push rbp
    mov rbp,rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12,QWORD PTR [rbp+16]      # K_len
    test r8,r8
    jz .done
    test r12,r12
    jz .done

    mov r13,r9
    shl r13,2                       # x row stride bytes
    mov r14,QWORD PTR [rbp+24]
    shl r14,2                       # y row stride bytes
    mov r15,rsi                     # fixed w chunk base

.main4:
    cmp r8,4
    jb .tailcheck

    cmp DWORD PTR [rbp+32],0
    je .load_y4
    vmovups ymm0,YMMWORD PTR [rdx]
    vmovups ymm1,YMMWORD PTR [rdx+32]
    vmovaps ymm2,ymm0
    vmovaps ymm3,ymm1
    vmovaps ymm4,ymm0
    vmovaps ymm5,ymm1
    vmovaps ymm6,ymm0
    vmovaps ymm7,ymm1
    jmp .init4_done
.load_y4:
    vmovups ymm0,YMMWORD PTR [rcx]
    vmovups ymm1,YMMWORD PTR [rcx+32]
    lea r10,[rcx+r14]
    vmovups ymm2,YMMWORD PTR [r10]
    vmovups ymm3,YMMWORD PTR [r10+32]
    lea r11,[r14+r14]
    lea r10,[rcx+r11]
    vmovups ymm4,YMMWORD PTR [r10]
    vmovups ymm5,YMMWORD PTR [r10+32]
    lea r11,[r14+r14*2]
    lea r10,[rcx+r11]
    vmovups ymm6,YMMWORD PTR [r10]
    vmovups ymm7,YMMWORD PTR [r10+32]
.init4_done:
    mov r10,rdi
    lea r11,[rdi+r13]
    lea rax,[r11+r13]
    lea rbx,[rax+r13]
    mov rsi,r15
    mov r9,r12
.k4:
    vmovups ymm9,YMMWORD PTR [rsi]
    vmovups ymm10,YMMWORD PTR [rsi+32]
    vbroadcastss ymm8,DWORD PTR [r10]
    vfmadd231ps ymm0,ymm8,ymm9
    vfmadd231ps ymm1,ymm8,ymm10
    vbroadcastss ymm8,DWORD PTR [r11]
    vfmadd231ps ymm2,ymm8,ymm9
    vfmadd231ps ymm3,ymm8,ymm10
    vbroadcastss ymm8,DWORD PTR [rax]
    vfmadd231ps ymm4,ymm8,ymm9
    vfmadd231ps ymm5,ymm8,ymm10
    vbroadcastss ymm8,DWORD PTR [rbx]
    vfmadd231ps ymm6,ymm8,ymm9
    vfmadd231ps ymm7,ymm8,ymm10
    add r10,4
    add r11,4
    add rax,4
    add rbx,4
    add rsi,64
    dec r9
    jnz .k4

    vmovups YMMWORD PTR [rcx],ymm0
    vmovups YMMWORD PTR [rcx+32],ymm1
    lea r10,[rcx+r14]
    vmovups YMMWORD PTR [r10],ymm2
    vmovups YMMWORD PTR [r10+32],ymm3
    lea r11,[r14+r14]
    lea r10,[rcx+r11]
    vmovups YMMWORD PTR [r10],ymm4
    vmovups YMMWORD PTR [r10+32],ymm5
    lea r11,[r14+r14*2]
    lea r10,[rcx+r11]
    vmovups YMMWORD PTR [r10],ymm6
    vmovups YMMWORD PTR [r10+32],ymm7

    lea r10,[r13+r13*2]
    add rdi,r10
    add rdi,r13                     # +4 x rows
    lea r10,[r14+r14*2]
    add rcx,r10
    add rcx,r14                     # +4 y rows
    sub r8,4
    jmp .main4

.tailcheck:
    test r8,r8
    jz .done
.tailrow:
    cmp DWORD PTR [rbp+32],0
    je .load_y1
    vmovups ymm0,YMMWORD PTR [rdx]
    vmovups ymm1,YMMWORD PTR [rdx+32]
    jmp .init1_done
.load_y1:
    vmovups ymm0,YMMWORD PTR [rcx]
    vmovups ymm1,YMMWORD PTR [rcx+32]
.init1_done:
    mov r10,rdi
    mov rsi,r15
    mov r9,r12
.k1:
    vmovups ymm9,YMMWORD PTR [rsi]
    vmovups ymm10,YMMWORD PTR [rsi+32]
    vbroadcastss ymm8,DWORD PTR [r10]
    vfmadd231ps ymm0,ymm8,ymm9
    vfmadd231ps ymm1,ymm8,ymm10
    add r10,4
    add rsi,64
    dec r9
    jnz .k1
    vmovups YMMWORD PTR [rcx],ymm0
    vmovups YMMWORD PTR [rcx+32],ymm1
    add rdi,r13
    add rcx,r14
    dec r8
    jnz .tailrow

.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
.size ds_linear_f32_avx2_n16_kblock_accum,.-ds_linear_f32_avx2_n16_kblock_accum
.section .note.GNU-stack,"",@progbits
