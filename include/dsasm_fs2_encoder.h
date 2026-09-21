#ifndef DSASM_FS2_ENCODER_H
#define DSASM_FS2_ENCODER_H
#include <stddef.h>
#include <stdint.h>
#include "dsasm_threadpool.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *ln1_gamma;             /* [C] */
    const float *ln1_beta;              /* [C] */
    const float *qkv_weight_m4n16;      /* packed16 [3C,C] */
    const float *qkv_bias;              /* [3C], zero for current upstream */
    const float *out_weight_m4n16;      /* packed16 [C,C] */
    const float *out_bias;              /* [C], zero for current upstream */
    const float *ln2_gamma;             /* [C] */
    const float *ln2_beta;              /* [C] */
    const float *ffn1_weight_m4n16;     /* packed16 [4C,3C] for Conv1d(k=3) */
    const float *ffn1_bias;             /* [4C] */
    const float *ffn2_weight_m4n16;     /* packed16 [C,4C] */
    const float *ffn2_bias;             /* [C] */
} DSAsmFS2EncoderLayer;

typedef struct {
    uint32_t vocab_size;
    uint32_t hidden_size;               /* current acoustic default: 384 */
    uint32_t num_layers;                /* current acoustic default: 4 */
    uint32_t num_heads;                 /* current acoustic default: 2 */
    uint32_t ffn_kernel_size;            /* M21 supports current upstream k=3 */
    uint32_t rope_interleaved;           /* current acoustic default: 0 */
    float rope_theta;                    /* current acoustic default: 10000 */
    const float *token_embedding;        /* [vocab,C], padding row 0 */
    const float *dur_weight;             /* AdamWLinear(1,C) weight squeezed -> [C] */
    const float *dur_bias;               /* [C] */
    const DSAsmFS2EncoderLayer *layers;  /* [num_layers] */
    const float *final_ln_gamma;         /* [C] */
    const float *final_ln_beta;          /* [C] */
} DSAsmFS2EncoderWeights;

/* Workspace size for B=1 token-level encoder forward. */
size_t ds_fs2_encoder_workspace_floats(const DSAsmFS2EncoderWeights *w,size_t tokens);

/* Current upstream FastSpeech2Encoder path for the default acoustic config:
   token embedding + log1p(duration) embedding -> 4-layer RoPE Transformer ->
   final LayerNorm. token_ids==0 are padding and are zeroed at the same points
   as upstream. Dropout is omitted because this is inference/eval only. */
int ds_fs2_encoder_forward_f32_avx2(
    const DSAsmFS2EncoderWeights *w,
    const int32_t *token_ids,const int32_t *durations,size_t tokens,
    float *out_tc,float *workspace,DSAsmThreadPool *pool);

/* M21 full current-default acoustic conditioner after the token Transformer.
   Optional language/speaker/variance/key-shift/speed branches remain outside
   this default-profile ABI; current configs/acoustic.yaml disables them. */
/* M25 deployment-only optional conditioner branches.  Kept separate from
   DSAsmFS2AcousticWeights so DSFS21/M21 ABI remains unchanged. */
typedef struct {
    uint32_t num_languages;             /* includes padding language row 0 */
    uint32_t flags;                     /* DSASM_FS2_FEAT_* */
    const float *language_embedding;    /* [num_languages,C] */
    const float *language_token_mask;   /* [vocab], 1 only for cross-lingual tokens */
    const float *breath_weight;         /* [C], Linear(1,C) squeezed */
    const float *breath_bias;           /* [C] */
    const float *voicing_weight;        /* [C] */
    const float *voicing_bias;          /* [C] */
    const float *tension_weight;        /* [C] */
    const float *tension_bias;          /* [C] */
    const float *key_shift_weight;      /* [C] */
    const float *key_shift_bias;        /* [C] */
    const float *speed_weight;          /* [C] */
    const float *speed_bias;            /* [C] */
    const float *stretch_table;         /* [1001,C], deployment-export lookup table */
    float breath_scale;
    float voicing_scale;
    float tension_scale;
    float gender_clip_min;
    float gender_clip_max;
    float gender_pre_scale;
    float key_shift_scale;
    float speed_clip_min;
    float speed_clip_max;
    float speed_scale;
} DSAsmFS2DeploymentExtras;

enum {
    DSASM_FS2_FEAT_LANGUAGE    = 1u<<0,
    DSASM_FS2_FEAT_BREATH     = 1u<<1,
    DSASM_FS2_FEAT_VOICING    = 1u<<2,
    DSASM_FS2_FEAT_TENSION    = 1u<<3,
    DSASM_FS2_FEAT_KEY_SHIFT  = 1u<<4,
    DSASM_FS2_FEAT_SPEED      = 1u<<5,
    DSASM_FS2_FEAT_SPEAKER    = 1u<<6,
    DSASM_FS2_FEAT_STRETCH_TABLE = 1u<<7,
    DSASM_FS2_FEAT_LANGUAGE_MASK = 1u<<8
};

typedef struct {
    const int32_t *languages;            /* [P], 0 for padding */
    const float *breathiness;            /* [T], optional -> 0 */
    const float *voicing;                /* [T], optional -> 0 */
    const float *tension;                /* [T], optional -> 0 */
    const float *gender;                 /* [T], optional -> 0 */
    const float *velocity;               /* [T], optional -> 1 */
    const float *speaker_embedding_tc;   /* [T,C], optional */
} DSAsmFS2DeploymentInputs;

typedef struct {
    DSAsmFS2EncoderWeights encoder;
    const float *stretch_w1_m4n16;      /* [4C,C] */
    const float *stretch_b1;            /* [4C] */
    const float *stretch_w2_m4n16;      /* [C,4C] */
    const float *stretch_b2;            /* [C] */
    const float *gru_w_ih_m4n16;        /* [3C,C], PyTorch gate order r,z,n */
    const float *gru_b_ih;               /* [3C] */
    const float *gru_w_hh_m4n16;        /* [3C,C] */
    const float *gru_b_hh;               /* [3C] */
    const float *pitch_weight;           /* Linear(1,C) squeezed [C] */
    const float *pitch_bias;             /* [C] */
} DSAsmFS2AcousticWeights;

size_t ds_fs2_acoustic_condition_workspace_floats(
    const DSAsmFS2AcousticWeights *w,size_t text_tokens,size_t mel_frames);

/* B=1 default acoustic FS2 path through condition generation:
   Transformer -> mel2ph gather -> stretch SinusoidalPosEmb/MLP -> GRU residual
   -> log1p(f0/700) pitch embedding. */
int ds_fs2_acoustic_condition_f32_avx2(
    const DSAsmFS2AcousticWeights *w,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    float *condition_tc,float *workspace,DSAsmThreadPool *pool);

/* M25 deployment path: same base conditioner plus language, variance,
   gender/key-shift, velocity/speed and frame-wise external speaker embedding. */
typedef struct {
    float *encoder_txt_pc;    /* [P,C] after FastSpeech2Encoder final mask */
    float *gathered_tc;       /* [T,C] after mel2ph gather, before stretch */
    float *stretch_tc;        /* [T,C] after stretch embedding residual */
    float *gru_tc;            /* [T,C] after stretch GRU residual */
    float *pitch_tc;          /* [T,C] after pitch embedding */
    float *variance_tc;       /* [T,C] after breathiness/voicing/tension */
    float *key_shift_tc;      /* [T,C] after gender/key-shift */
    float *speed_tc;          /* [T,C] after velocity/speed */
    float *speaker_tc;        /* [T,C] final condition after speaker add */
} DSAsmFS2DeploymentDebug;

int ds_fs2_acoustic_condition_deploy_debug_f32_avx2(
    const DSAsmFS2AcousticWeights*w,const DSAsmFS2DeploymentExtras*x,const DSAsmFS2DeploymentInputs*in,
    const int32_t*tok,size_t P,const int32_t*mel2ph,const float*f0,size_t T,
    float*cond,float*ws,DSAsmThreadPool*pool,DSAsmFS2DeploymentDebug*dbg);

int ds_fs2_acoustic_condition_deploy_f32_avx2(
    const DSAsmFS2AcousticWeights *w,const DSAsmFS2DeploymentExtras *x,
    const DSAsmFS2DeploymentInputs *in,
    const int32_t *token_ids,size_t text_tokens,
    const int32_t *mel2ph,const float *f0,size_t mel_frames,
    float *condition_tc,float *workspace,DSAsmThreadPool *pool);

#ifdef __cplusplus
}
#endif
#endif
