#include "privacy_filter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static char *make_long_text(int words) {
    const char *piece = "hello ";
    size_t piece_len = strlen(piece);
    size_t len = (size_t)words * piece_len + 64;
    char *text = (char *)malloc(len);
    if (!text) return NULL;
    size_t pos = 0;
    for (int i = 0; i < words; ++i) {
        memcpy(text + pos, piece, piece_len);
        pos += piece_len;
    }
    memcpy(text + pos, "alice@example.com 555-123-4567", 30);
    text[pos + 30] = 0;
    return text;
}

static void usage(FILE *out) {
    fputs("usage: pf_bench [--model DIR] [--repeat N] [--long WORDS] [text]\n", out);
}

int main(int argc, char **argv) {
    const char *model_dir = "model";
    const char *text = "Contact Alice at alice@example.com or call 555-123-4567.";
    char *generated_text = NULL;
    int repeat = 3;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--repeat") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            repeat = atoi(argv[++i]);
            if (repeat <= 0) repeat = 1;
        } else if (strcmp(argv[i], "--long") == 0) {
            if (i + 1 >= argc) {
                usage(stderr);
                return 2;
            }
            free(generated_text);
            generated_text = make_long_text(atoi(argv[++i]));
            if (!generated_text) return 1;
            text = generated_text;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            free(generated_text);
            usage(stdout);
            return 0;
        } else {
            text = argv[i];
        }
    }

    double t0 = now_seconds();
    PFModel *model = pf_load_model(model_dir);
    double t1 = now_seconds();
    if (!model) {
        free(generated_text);
        return 1;
    }

    PFToken *tokens = NULL;
    int token_count = 0;
    if (pf_tokenize(model, text, &tokens, &token_count) != 0) {
        free(generated_text);
        pf_free_model(model);
        return 1;
    }
    pf_free(tokens);

    double best = 1.0e100;
    int last_count = 0;
    for (int i = 0; i < repeat; ++i) {
        PFSpan *spans = NULL;
        int count = 0;
        double a = now_seconds();
        int rc = pf_predict(model, text, &spans, &count);
        double b = now_seconds();
        if (rc != 0) {
            free(generated_text);
            pf_free_model(model);
            return 1;
        }
        if (b - a < best) best = b - a;
        last_count = count;
        pf_free(spans);
    }

    printf("load_seconds=%.6f\n", t1 - t0);
    printf("tokens=%d\n", token_count);
    printf("spans=%d\n", last_count);
    printf("best_predict_seconds=%.6f\n", best);
    printf("tokens_per_second=%.2f\n", best > 0.0 ? (double)token_count / best : 0.0);

    pf_free_model(model);
    free(generated_text);
    return 0;
}
