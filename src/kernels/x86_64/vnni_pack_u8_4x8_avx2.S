.intel_syntax noprefix
.text
.globl ds_vnni_pack_u8_4x8_avx2
.type ds_vnni_pack_u8_4x8_avx2,@function
# void ds_vnni_pack_u8_4x8_avx2(
#   const uint8_t *qpad, uint8_t *xpack, const int32_t *offset4,
#   size_t K4, size_t tb0, size_t tb1)
# Each output block is 8 time lanes x 4 reduction bytes, lane-major.
ds_vnni_pack_u8_4x8_avx2:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, r8              # tb
    mov r13, r9              # tb1
    cmp r12, r13
    jae .Ldone

    # xpack points at the full buffer; advance to tb0 block.
    mov rax, r12
    imul rax, rcx
    shl rax, 5               # * K4 * 32
    add rsi, rax

.Ltb:
    mov r14, r12
    shl r14, 3               # time base = tb*8
    mov r15, rdx             # offset map restart for every tb
    xor ebx, ebx              # kb=0
.Lkb:
    movsxd rax, DWORD PTR [r15]
    lea r10, [rdi + r14]
    movq xmm0, QWORD PTR [r10 + rax]
    movsxd rax, DWORD PTR [r15 + 4]
    movq xmm1, QWORD PTR [r10 + rax]
    movsxd rax, DWORD PTR [r15 + 8]
    movq xmm2, QWORD PTR [r10 + rax]
    movsxd rax, DWORD PTR [r15 + 12]
    movq xmm3, QWORD PTR [r10 + rax]

    # [a0..a7],[b0..b7] -> a0,b0,a1,b1,...
    punpcklbw xmm0, xmm1
    punpcklbw xmm2, xmm3
    movdqa xmm4, xmm0
    punpcklwd xmm0, xmm2     # lanes 0..3: a,b,c,d quartets
    punpckhwd xmm4, xmm2     # lanes 4..7
    movdqu XMMWORD PTR [rsi], xmm0
    movdqu XMMWORD PTR [rsi + 16], xmm4

    add r15, 16
    add rsi, 32
    inc rbx
    cmp rbx, rcx
    jb .Lkb

    inc r12
    cmp r12, r13
    jb .Ltb

.Ldone:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
.size ds_vnni_pack_u8_4x8_avx2, .-ds_vnni_pack_u8_4x8_avx2
.section .note.GNU-stack,"",@progbits
