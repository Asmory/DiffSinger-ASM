.intel_syntax noprefix
.text

# void ds_add3_broadcast_f32_avx2(
#   const float *a, const float *b, const float *c,
#   float *y, size_t M, size_t N);
# y[m,n] = a[m,n] + b[m,n] + c[n].  y may alias a or b.
.globl ds_add3_broadcast_f32_avx2
.type ds_add3_broadcast_f32_avx2,@function
ds_add3_broadcast_f32_avx2:
    test r8,r8
    jz .done
    test r9,r9
    jz .done
    mov r10,r9
    and r10,-8
.row:
    xor rax,rax
.vec:
    cmp rax,r10
    jae .tail
    vmovups ymm0,YMMWORD PTR [rdi+rax*4]
    vaddps ymm0,ymm0,YMMWORD PTR [rsi+rax*4]
    vaddps ymm0,ymm0,YMMWORD PTR [rdx+rax*4]
    vmovups YMMWORD PTR [rcx+rax*4],ymm0
    add rax,8
    jmp .vec
.tail:
    cmp rax,r9
    jae .next
.tail_loop:
    vmovss xmm0,DWORD PTR [rdi+rax*4]
    vaddss xmm0,xmm0,DWORD PTR [rsi+rax*4]
    vaddss xmm0,xmm0,DWORD PTR [rdx+rax*4]
    vmovss DWORD PTR [rcx+rax*4],xmm0
    inc rax
    cmp rax,r9
    jb .tail_loop
.next:
    lea r11,[r9*4]
    add rdi,r11
    add rsi,r11
    add rcx,r11
    dec r8
    jnz .row
.done:
    vzeroupper
    ret
.size ds_add3_broadcast_f32_avx2,.-ds_add3_broadcast_f32_avx2
.section .note.GNU-stack,"",@progbits
