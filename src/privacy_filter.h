#ifndef PRIVACY_FILTER_H
#define PRIVACY_FILTER_H

#include <stddef.h>
#include <stdint.h>

#define PF_HIDDEN 640
#define PF_INTERMEDIATE 640
#define PF_LAYERS 8
#define PF_HEADS 14
#define PF_KV_HEADS 2
#define PF_HEAD_DIM 64
#define PF_EXPERTS 128
#define PF_TOPK 4
#define PF_LABELS 33
#define PF_SLIDING_WINDOW 128
#define PF_VOCAB_SIZE 200064
#define PF_PAD_TOKEN 199999
#define PF_ROPE_THETA 150000.0f
#define PF_ROPE_FACTOR 32.0f
#define PF_ROPE_ORIGINAL_MAX 4096
#define PF_ROPE_BETA_FAST 32.0f
#define PF_ROPE_BETA_SLOW 1.0f
#define PF_RMS_EPS 1.0e-5f
#define PF_SWIGLU_ALPHA 1.702f
#define PF_SWIGLU_LIMIT 7.0f

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    uint32_t start;
    uint32_t end;
} PFToken;

typedef struct {
    char label[32];
    uint32_t start;
    uint32_t end;
} PFSpan;

typedef struct PFModel PFModel;

PFModel *pf_load_model(const char *model_dir);
void pf_free_model(PFModel *model);
int pf_tokenize(PFModel *model, const char *text, PFToken **tokens_out, int *count_out);
int pf_predict(PFModel *model, const char *text, PFSpan **spans_out, int *count_out);
int pf_predict_raw(PFModel *model, const char *text, PFSpan **spans_out, int *count_out);
int pf_redact(PFModel *model, const char *text, char **redacted_out);
void pf_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
