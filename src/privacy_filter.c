#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "privacy_filter.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef PF_USE_AVX2
#include <immintrin.h>

static inline __m256 pf_load_bf16x8(const uint8_t *p) {
    __m128i bf16 = _mm_loadu_si128((const __m128i *)p);
    __m256i i32 = _mm256_cvtepu16_epi32(bf16);
    return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

static inline float pf_hsum_f32x8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PF_MAX_TOKEN_BYTES 256
#define PF_NEG_INF (-1.0e30f)
#define PF_PARAGRAPH_THRESHOLD 256

static const char *PF_SPAN_NAMES[9] = {
    "O", "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
};

static const uint8_t PF_TOKEN_TO_SPAN[PF_LABELS] = {
    0,
    1,1,1,1,
    2,2,2,2,
    3,3,3,3,
    4,4,4,4,
    5,5,5,5,
    6,6,6,6,
    7,7,7,7,
    8,8,8,8,
};

static const char PF_TOKEN_BOUNDARY[PF_LABELS] = {
    0,
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
    'B','I','E','S',
};

typedef struct {
    int id;
    uint16_t len;
    uint8_t bytes[PF_MAX_TOKEN_BYTES];
} VocabEntry;

typedef struct TrieNode {
    int token_id;
    struct TrieNode *child[256];
} TrieNode;

typedef struct {
    char *name;
    uint32_t dtype;
    uint32_t ndim;
    uint64_t dim[4];
    uint64_t offset;
    uint64_t nbytes;
    void *data;
} Tensor;

struct PFModel {
    char *model_dir;
    char *weights_path;
    FILE *weights_file;
    uint8_t *weights_map;
    uint64_t weights_map_size;
#ifdef _WIN32
    HANDLE weights_handle;
    HANDLE weights_mapping;
#endif
    Tensor *tensors;
    int tensor_count;
    TrieNode *trie;
    VocabEntry *id_to_vocab[PF_VOCAB_SIZE];
    VocabEntry *specials;
    int special_count;
    float *x;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *moe;
    float *router_logits;
    int *moe_expert;
    int *moe_token;
    int *moe_sorted_token;
    float *moe_weight;
    float *moe_sorted_weight;
    float *moe_thread_scratch;
    size_t moe_thread_scratch_count;
    int moe_assignments;
    int moe_scratch_threads;
    float *rope_cos;
    float *rope_sin;
    float *logits;
    float *transition;
    float *start_scores;
    float *end_scores;
    int buffer_tokens;
    int rope_tokens;
    float transition_bias_background_stay;
    float transition_bias_background_to_start;
    float transition_bias_inside_to_continue;
    float transition_bias_inside_to_end;
    float transition_bias_end_to_background;
    float transition_bias_end_to_start;
};

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return p;
}

static void *xmalloc(size_t size) {
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)xmalloc(n);
    memcpy(out, s, n);
    return out;
}

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int checked_mul3_size(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab;
    if (checked_mul_size(a, b, &ab) != 0) return -1;
    return checked_mul_size(ab, c, out);
}

static void *xreallocarray(void *ptr, size_t n, size_t size) {
    size_t bytes;
    if (checked_mul_size(n, size, &bytes) != 0) {
        fprintf(stderr, "size overflow\n");
        exit(2);
    }
    void *next = realloc(ptr, bytes);
    if (!next && bytes != 0) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return next;
}

static int grow_float_buffer(float **ptr, size_t count) {
    size_t bytes;
    if (checked_mul_size(count, sizeof(float), &bytes) != 0) return -1;
    float *next = (float *)realloc(*ptr, bytes);
    if (!next && bytes != 0) return -1;
    *ptr = next;
    return 0;
}

static int grow_int_buffer(int **ptr, size_t count) {
    size_t bytes;
    if (checked_mul_size(count, sizeof(int), &bytes) != 0) return -1;
    int *next = (int *)realloc(*ptr, bytes);
    if (!next && bytes != 0) return -1;
    *ptr = next;
    return 0;
}

static char *join_path(const char *dir, const char *name) {
    size_t a = strlen(dir);
    size_t b = strlen(name);
    int need_sep = a > 0 && dir[a - 1] != '/' && dir[a - 1] != '\\';
    char *out = (char *)xmalloc(a + b + (need_sep ? 2 : 1));
    memcpy(out, dir, a);
    if (need_sep) out[a++] = '/';
    memcpy(out + a, name, b + 1);
    return out;
}

static TrieNode *trie_node_new(void) {
    TrieNode *n = (TrieNode *)xcalloc(1, sizeof(TrieNode));
    n->token_id = -1;
    return n;
}

static void trie_insert(TrieNode *root, const uint8_t *bytes, int len, int token_id) {
    TrieNode *cur = root;
    for (int i = 0; i < len; ++i) {
        uint8_t b = bytes[i];
        if (!cur->child[b]) cur->child[b] = trie_node_new();
        cur = cur->child[b];
    }
    cur->token_id = token_id;
}

static void trie_free(TrieNode *n) {
    if (!n) return;
    for (int i = 0; i < 256; ++i) trie_free(n->child[i]);
    free(n);
}

static char *read_file_bytes(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)xmalloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    buf[n] = 0;
    *len_out = (size_t)n;
    return buf;
}

static uint64_t read_le64_file(FILE *f) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) return 0;
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
    return v;
}

static const char *skip_ws_json(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}

static const char *find_bytes(const char *p, const char *end, const char *needle, size_t needle_len) {
    if (needle_len == 0) return p;
    for (; p + needle_len <= end; ++p) {
        if (*p == *needle && memcmp(p, needle, needle_len) == 0) return p;
    }
    return NULL;
}

static const char *find_json_key(const char *p, const char *end, const char *key) {
    char pattern[128];
    int n = snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    if (n <= 0 || (size_t)n >= sizeof(pattern)) return NULL;
    const char *q = p;
    while ((q = find_bytes(q, end, pattern, (size_t)n)) != NULL) {
        const char *v = skip_ws_json(q + n, end);
        if (v < end && *v == ':') return skip_ws_json(v + 1, end);
        q += n;
    }
    return NULL;
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_hex4(const char *p, const char *end, uint32_t *cp) {
    if (p + 4 > end) return -1;
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        int h = hex_value(p[i]);
        if (h < 0) return -1;
        v = (v << 4) | (uint32_t)h;
    }
    *cp = v;
    return 0;
}

static int utf8_next(const char **pp, const char *end, uint32_t *cp) {
    const unsigned char *p = (const unsigned char *)*pp;
    if ((const char *)p >= end) return -1;
    if (p[0] < 0x80) {
        *cp = p[0];
        *pp = (const char *)(p + 1);
        return 0;
    }
    if ((p[0] & 0xE0) == 0xC0 && (const char *)(p + 2) <= end) {
        *cp = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
        *pp = (const char *)(p + 2);
        return 0;
    }
    if ((p[0] & 0xF0) == 0xE0 && (const char *)(p + 3) <= end) {
        *cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
        *pp = (const char *)(p + 3);
        return 0;
    }
    if ((p[0] & 0xF8) == 0xF0 && (const char *)(p + 4) <= end) {
        *cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12) | ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
        *pp = (const char *)(p + 4);
        return 0;
    }
    return -1;
}

static int json_string_next_codepoint(const char **pp, const char *end, uint32_t *cp) {
    const char *p = *pp;
    if (p >= end) return -1;
    if (*p != '\\') return utf8_next(pp, end, cp);
    ++p;
    if (p >= end) return -1;
    switch (*p++) {
        case '"': *cp = '"'; break;
        case '\\': *cp = '\\'; break;
        case '/': *cp = '/'; break;
        case 'b': *cp = '\b'; break;
        case 'f': *cp = '\f'; break;
        case 'n': *cp = '\n'; break;
        case 'r': *cp = '\r'; break;
        case 't': *cp = '\t'; break;
        case 'u': {
            uint32_t hi;
            if (parse_hex4(p, end, &hi) != 0) return -1;
            p += 4;
            if (hi >= 0xD800 && hi <= 0xDBFF && p + 6 <= end && p[0] == '\\' && p[1] == 'u') {
                uint32_t lo;
                if (parse_hex4(p + 2, end, &lo) != 0) return -1;
                if (lo >= 0xDC00 && lo <= 0xDFFF) {
                    hi = 0x10000u + ((hi - 0xD800u) << 10) + (lo - 0xDC00u);
                    p += 6;
                }
            }
            *cp = hi;
            break;
        }
        default:
            return -1;
    }
    *pp = p;
    return 0;
}

static void append_utf8(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    char tmp[4];
    int n = 0;
    if (cp <= 0x7F) {
        tmp[n++] = (char)cp;
    } else if (cp <= 0x7FF) {
        tmp[n++] = (char)(0xC0 | (cp >> 6));
        tmp[n++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        tmp[n++] = (char)(0xE0 | (cp >> 12));
        tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[n++] = (char)(0x80 | (cp & 0x3F));
    } else {
        tmp[n++] = (char)(0xF0 | (cp >> 18));
        tmp[n++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        tmp[n++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        tmp[n++] = (char)(0x80 | (cp & 0x3F));
    }
    if (*len + (size_t)n + 1 > *cap) {
        *cap *= 2;
        if (*len + (size_t)n + 1 > *cap) *cap = *len + (size_t)n + 1;
        char *next = (char *)xreallocarray(*buf, *cap, 1);
        if (!next) {
            free(*buf);
            fprintf(stderr, "out of memory\n");
            exit(2);
        }
        *buf = next;
    }
    memcpy(*buf + *len, tmp, (size_t)n);
    *len += (size_t)n;
    (*buf)[*len] = 0;
}

static char *parse_json_string_alloc(const char **pp, const char *end) {
    const char *p = skip_ws_json(*pp, end);
    if (p >= end || *p != '"') return NULL;
    ++p;
    size_t cap = 64, len = 0;
    char *out = (char *)xmalloc(cap);
    out[0] = 0;
    while (p < end && *p != '"') {
        uint32_t cp;
        if (json_string_next_codepoint(&p, end, &cp) != 0) {
            free(out);
            return NULL;
        }
        append_utf8(&out, &len, &cap, cp);
    }
    if (p >= end || *p != '"') {
        free(out);
        return NULL;
    }
    *pp = p + 1;
    return out;
}

static const char *json_skip_string_raw(const char *p, const char *end) {
    if (p >= end || *p != '"') return NULL;
    ++p;
    while (p < end) {
        if (*p == '\\') {
            p += 2;
        } else if (*p == '"') {
            return p + 1;
        } else {
            ++p;
        }
    }
    return NULL;
}

static const char *json_skip_value(const char *p, const char *end) {
    p = skip_ws_json(p, end);
    if (p >= end) return NULL;
    if (*p == '"') return json_skip_string_raw(p, end);
    if (*p == '{' || *p == '[') {
        char open = *p;
        char close = open == '{' ? '}' : ']';
        int depth = 1;
        ++p;
        while (p < end && depth > 0) {
            if (*p == '"') {
                p = json_skip_string_raw(p, end);
                if (!p) return NULL;
            } else if (*p == open) {
                ++depth;
                ++p;
            } else if (*p == close) {
                --depth;
                ++p;
            } else {
                ++p;
            }
        }
        return depth == 0 ? p : NULL;
    }
    while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    return p;
}

static uint64_t parse_json_uint(const char **pp, const char *end, int *ok) {
    const char *p = skip_ws_json(*pp, end);
    uint64_t v = 0;
    int any = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        any = 1;
        v = v * 10u + (uint64_t)(*p - '0');
        ++p;
    }
    *pp = p;
    *ok = any;
    return v;
}

static double parse_json_number(const char **pp, const char *end, int *ok) {
    const char *p = skip_ws_json(*pp, end);
    if (p >= end) {
        *ok = 0;
        return 0.0;
    }
    errno = 0;
    char *q = NULL;
    double v = strtod(p, &q);
    if (q == p || q > end || errno == ERANGE) {
        *ok = 0;
        return 0.0;
    }
    *pp = q;
    *ok = 1;
    return v;
}

static int parse_json_bool(const char *p, const char *end, int *value) {
    p = skip_ws_json(p, end);
    if (p + 4 <= end && memcmp(p, "true", 4) == 0) {
        *value = 1;
        return 0;
    }
    if (p + 5 <= end && memcmp(p, "false", 5) == 0) {
        *value = 0;
        return 0;
    }
    return -1;
}

static int byte_level_codepoint_to_byte(uint32_t cp) {
    static int map[512];
    static int initialized = 0;
    if (!initialized) {
        for (int i = 0; i < 512; ++i) map[i] = -1;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
                map[b] = b;
            } else {
                map[256 + n] = b;
                ++n;
            }
        }
        initialized = 1;
    }
    if (cp >= 512) return -1;
    return map[cp];
}

static int parse_token_bytes(const char **pp, const char *end, uint8_t *bytes, uint16_t *len_out) {
    const char *p = skip_ws_json(*pp, end);
    if (p >= end || *p != '"') return -1;
    ++p;
    int len = 0;
    while (p < end && *p != '"') {
        uint32_t cp;
        if (json_string_next_codepoint(&p, end, &cp) != 0) return -1;
        int b = byte_level_codepoint_to_byte(cp);
        if (b < 0 || len >= PF_MAX_TOKEN_BYTES) return -1;
        bytes[len++] = (uint8_t)b;
    }
    if (p >= end || *p != '"') return -1;
    *pp = p + 1;
    *len_out = (uint16_t)len;
    return 0;
}

static int add_vocab_entry(PFModel *m, uint32_t id, const uint8_t *bytes, uint16_t len) {
    if (id >= PF_VOCAB_SIZE || len > PF_MAX_TOKEN_BYTES) return -1;
    VocabEntry *e = (VocabEntry *)xcalloc(1, sizeof(VocabEntry));
    e->id = (int)id;
    e->len = len;
    memcpy(e->bytes, bytes, len);
    if (m->id_to_vocab[id]) free(m->id_to_vocab[id]);
    m->id_to_vocab[id] = e;
    trie_insert(m->trie, e->bytes, e->len, e->id);
    return 0;
}

static int add_special_entry(PFModel *m, int *cap, uint32_t id, const char *content) {
    if (id >= PF_VOCAB_SIZE) return -1;
    size_t len = strlen(content);
    if (len > PF_MAX_TOKEN_BYTES) return -1;
    if (m->special_count == *cap) {
        *cap *= 2;
        VocabEntry *next = (VocabEntry *)xreallocarray(m->specials, (size_t)(*cap), sizeof(VocabEntry));
        if (!next) return -1;
        m->specials = next;
    }
    VocabEntry *e = &m->specials[m->special_count++];
    memset(e, 0, sizeof(*e));
    e->id = (int)id;
    e->len = (uint16_t)len;
    memcpy(e->bytes, content, len);
    return 0;
}

static int load_vocab(PFModel *m, const char *path) {
    size_t json_len = 0;
    char *json = read_file_bytes(path, &json_len);
    if (!json) return -1;
    const char *begin = json;
    const char *end = json + json_len;

    const char *ignore = find_json_key(begin, end, "ignore_merges");
    int ignore_merges = 0;
    if (!ignore || parse_json_bool(ignore, end, &ignore_merges) != 0 || !ignore_merges) {
        fprintf(stderr, "tokenizer.json must be BPE with ignore_merges=true\n");
        free(json);
        return -1;
    }

    const char *vocab = find_json_key(begin, end, "vocab");
    if (!vocab || *skip_ws_json(vocab, end) != '{') {
        fprintf(stderr, "tokenizer.json missing model.vocab\n");
        free(json);
        return -1;
    }
    const char *p = skip_ws_json(vocab, end) + 1;
    m->trie = trie_node_new();
    while (1) {
        p = skip_ws_json(p, end);
        if (p >= end) {
            free(json);
            return -1;
        }
        if (*p == '}') {
            ++p;
            break;
        }
        uint8_t token_bytes[PF_MAX_TOKEN_BYTES];
        uint16_t token_len = 0;
        if (parse_token_bytes(&p, end, token_bytes, &token_len) != 0) {
            fprintf(stderr, "failed to parse tokenizer vocab token\n");
            free(json);
            return -1;
        }
        p = skip_ws_json(p, end);
        if (p >= end || *p != ':') {
            free(json);
            return -1;
        }
        ++p;
        int ok = 0;
        uint64_t id = parse_json_uint(&p, end, &ok);
        if (!ok || add_vocab_entry(m, (uint32_t)id, token_bytes, token_len) != 0) {
            fprintf(stderr, "invalid tokenizer vocab entry\n");
            free(json);
            return -1;
        }
        p = skip_ws_json(p, end);
        if (p < end && *p == ',') ++p;
    }

    int special_cap = 64;
    m->specials = (VocabEntry *)xcalloc((size_t)special_cap, sizeof(VocabEntry));
    const char *added = find_json_key(begin, end, "added_tokens");
    if (added) {
        p = skip_ws_json(added, end);
        if (p >= end || *p != '[') {
            free(json);
            return -1;
        }
        ++p;
        while (1) {
            p = skip_ws_json(p, end);
            if (p >= end) {
                free(json);
                return -1;
            }
            if (*p == ']') break;
            if (*p != '{') {
                free(json);
                return -1;
            }
            const char *obj_start = p;
            const char *obj_end = json_skip_value(p, end);
            if (!obj_end) {
                free(json);
                return -1;
            }
            const char *idp = find_json_key(obj_start, obj_end, "id");
            const char *contentp = find_json_key(obj_start, obj_end, "content");
            const char *specialp = find_json_key(obj_start, obj_end, "special");
            int is_special = 0;
            if (idp && contentp && specialp && parse_json_bool(specialp, obj_end, &is_special) == 0 && is_special) {
                int ok = 0;
                uint64_t id = parse_json_uint(&idp, obj_end, &ok);
                char *content = parse_json_string_alloc(&contentp, obj_end);
                if (!ok || !content || add_special_entry(m, &special_cap, (uint32_t)id, content) != 0) {
                    free(content);
                    free(json);
                    return -1;
                }
                free(content);
            }
            p = skip_ws_json(obj_end, end);
            if (p < end && *p == ',') ++p;
        }
    }

    free(json);
    return 0;
}

static int parse_json_u64_array(const char *p, const char *end, uint64_t *out, int *count_out, int max_count) {
    p = skip_ws_json(p, end);
    if (p >= end || *p != '[') return -1;
    ++p;
    int count = 0;
    while (1) {
        p = skip_ws_json(p, end);
        if (p >= end) return -1;
        if (*p == ']') {
            *count_out = count;
            return 0;
        }
        if (count >= max_count) return -1;
        int ok = 0;
        out[count++] = parse_json_uint(&p, end, &ok);
        if (!ok) return -1;
        p = skip_ws_json(p, end);
        if (p < end && *p == ',') ++p;
    }
}

static int add_tensor_metadata(PFModel *m, int *cap, char *name, uint32_t dtype, int ndim, const uint64_t *dims, uint64_t offset, uint64_t nbytes) {
    if (m->tensor_count == *cap) {
        *cap *= 2;
        Tensor *next = (Tensor *)xreallocarray(m->tensors, (size_t)(*cap), sizeof(Tensor));
        if (!next) return -1;
        m->tensors = next;
    }
    Tensor *t = &m->tensors[m->tensor_count++];
    memset(t, 0, sizeof(*t));
    t->name = name;
    t->dtype = dtype;
    t->ndim = (uint32_t)ndim;
    for (int i = 0; i < 4; ++i) t->dim[i] = i < ndim ? dims[i] : 1;
    t->offset = offset;
    t->nbytes = nbytes;
    return 0;
}

static int parse_tensor_entry(PFModel *m, int *cap, char *name, const char *obj_start, const char *obj_end, uint64_t data_start) {
    const char *dtypep = find_json_key(obj_start, obj_end, "dtype");
    const char *shapep = find_json_key(obj_start, obj_end, "shape");
    const char *offsetsp = find_json_key(obj_start, obj_end, "data_offsets");
    if (!dtypep || !shapep || !offsetsp) return -1;
    char *dtype_s = parse_json_string_alloc(&dtypep, obj_end);
    if (!dtype_s) return -1;
    uint32_t dtype = 0;
    if (strcmp(dtype_s, "BF16") == 0) dtype = 1;
    else if (strcmp(dtype_s, "F32") == 0) dtype = 2;
    free(dtype_s);
    if (!dtype) return -1;

    uint64_t shape[4] = {1, 1, 1, 1};
    uint64_t offsets[2] = {0, 0};
    int ndim = 0;
    int noff = 0;
    if (parse_json_u64_array(shapep, obj_end, shape, &ndim, 4) != 0) return -1;
    if (parse_json_u64_array(offsetsp, obj_end, offsets, &noff, 2) != 0 || noff != 2 || offsets[1] < offsets[0]) return -1;
    return add_tensor_metadata(m, cap, name, dtype, ndim, shape, data_start + offsets[0], offsets[1] - offsets[0]);
}

static int load_index(PFModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open %s\n", path);
        return -1;
    }
    uint64_t header_len = read_le64_file(f);
    if (header_len == 0 || header_len > (uint64_t)SIZE_MAX - 1u) {
        fclose(f);
        fprintf(stderr, "invalid safetensors header: %s\n", path);
        return -1;
    }
    char *header = (char *)xmalloc((size_t)header_len + 1);
    if (fread(header, 1, (size_t)header_len, f) != (size_t)header_len) {
        fclose(f);
        free(header);
        return -1;
    }
    fclose(f);
    header[header_len] = 0;
    const char *begin = header;
    const char *end = header + header_len;
    const char *p = skip_ws_json(begin, end);
    if (p >= end || *p != '{') {
        free(header);
        return -1;
    }
    ++p;
    int cap = 256;
    m->tensors = (Tensor *)xcalloc((size_t)cap, sizeof(Tensor));
    uint64_t data_start = 8u + header_len;
    while (1) {
        p = skip_ws_json(p, end);
        if (p >= end) {
            free(header);
            return -1;
        }
        if (*p == '}') break;
        char *name = parse_json_string_alloc(&p, end);
        if (!name) {
            free(header);
            return -1;
        }
        p = skip_ws_json(p, end);
        if (p >= end || *p != ':') {
            free(name);
            free(header);
            return -1;
        }
        ++p;
        const char *obj_start = skip_ws_json(p, end);
        const char *obj_end = json_skip_value(obj_start, end);
        if (!obj_end) {
            free(name);
            free(header);
            return -1;
        }
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
        } else {
            if (parse_tensor_entry(m, &cap, name, obj_start, obj_end, data_start) != 0) {
                fprintf(stderr, "failed to parse tensor metadata for %s\n", name);
                free(name);
                free(header);
                return -1;
            }
        }
        p = skip_ws_json(obj_end, end);
        if (p < end && *p == ',') ++p;
    }
    free(header);
    return 0;
}

static int load_viterbi_calibration(PFModel *m, const char *path) {
    size_t len = 0;
    char *json = read_file_bytes(path, &len);
    if (!json) return 0;
    const char *begin = json;
    const char *end = json + len;
    struct BiasField { const char *key; float *dst; } fields[] = {
        {"transition_bias_background_stay", &m->transition_bias_background_stay},
        {"transition_bias_background_to_start", &m->transition_bias_background_to_start},
        {"transition_bias_inside_to_continue", &m->transition_bias_inside_to_continue},
        {"transition_bias_inside_to_end", &m->transition_bias_inside_to_end},
        {"transition_bias_end_to_background", &m->transition_bias_end_to_background},
        {"transition_bias_end_to_start", &m->transition_bias_end_to_start},
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        const char *p = find_json_key(begin, end, fields[i].key);
        if (!p) continue;
        int ok = 0;
        double v = parse_json_number(&p, end, &ok);
        if (!ok) {
            fprintf(stderr, "invalid viterbi calibration value: %s\n", fields[i].key);
            free(json);
            return -1;
        }
        *fields[i].dst = (float)v;
    }
    free(json);
    return 0;
}

static int tensor_compare(const void *a, const void *b) {
    const Tensor *ta = (const Tensor *)a;
    const Tensor *tb = (const Tensor *)b;
    return strcmp(ta->name, tb->name);
}

static Tensor *find_tensor_optional(PFModel *m, const char *name) {
    int lo = 0, hi = m->tensor_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(name, m->tensors[mid].name);
        if (c == 0) return &m->tensors[mid];
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return NULL;
}

static Tensor *find_tensor(PFModel *m, const char *name) {
    Tensor *t = find_tensor_optional(m, name);
    if (!t) fprintf(stderr, "missing tensor: %s\n", name);
    return t;
}

static uint32_t float_to_u32(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

static float u32_to_float(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof(x));
    return x;
}

static float bf16_to_float(uint16_t b) {
    return u32_to_float((uint32_t)b << 16);
}

static uint16_t read_le16_mem(const uint8_t *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static float tensor_get(Tensor *t, uint64_t idx) {
    if (t->dtype == 2) {
        return ((float *)t->data)[idx];
    }
    const uint8_t *p = (const uint8_t *)t->data + idx * 2;
    return bf16_to_float(read_le16_mem(p));
}

static void copy_tensor_row(float *out, Tensor *t, uint64_t row, int width) {
    uint64_t base = row * (uint64_t)width;
    if (t->dtype == 2) {
        memcpy(out, (const float *)t->data + base, (size_t)width * sizeof(float));
        return;
    }
    const uint8_t *p = (const uint8_t *)t->data + base * 2u;
#ifdef PF_USE_AVX2
    int i = 0;
    for (; i + 7 < width; i += 8) _mm256_storeu_ps(out + i, pf_load_bf16x8(p + (size_t)i * 2u));
    for (; i < width; ++i) out[i] = bf16_to_float(read_le16_mem(p + (size_t)i * 2u));
#else
    for (int i = 0; i < width; ++i) out[i] = bf16_to_float(read_le16_mem(p + (size_t)i * 2u));
#endif
}

static void unmap_weights(PFModel *m);

static int map_weights(PFModel *m) {
#ifdef _WIN32
    m->weights_handle = CreateFileA(m->weights_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m->weights_handle == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER size;
    if (!GetFileSizeEx(m->weights_handle, &size) || size.QuadPart <= 0) {
        unmap_weights(m);
        return -1;
    }
    m->weights_mapping = CreateFileMappingA(m->weights_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->weights_mapping) {
        unmap_weights(m);
        return -1;
    }
    m->weights_map = (uint8_t *)MapViewOfFile(m->weights_mapping, FILE_MAP_READ, 0, 0, 0);
    if (!m->weights_map) {
        unmap_weights(m);
        return -1;
    }
    m->weights_map_size = (uint64_t)size.QuadPart;
    return 0;
#else
    int fd = open(m->weights_path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return -1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return -1;
    m->weights_map = (uint8_t *)p;
    m->weights_map_size = (uint64_t)st.st_size;
    return 0;
#endif
}

static void unmap_weights(PFModel *m) {
#ifdef _WIN32
    if (m->weights_map) UnmapViewOfFile(m->weights_map);
    if (m->weights_mapping) CloseHandle(m->weights_mapping);
    if (m->weights_handle && m->weights_handle != INVALID_HANDLE_VALUE) CloseHandle(m->weights_handle);
    m->weights_handle = NULL;
    m->weights_mapping = NULL;
#else
    if (m->weights_map) munmap(m->weights_map, (size_t)m->weights_map_size);
#endif
    m->weights_map = NULL;
    m->weights_map_size = 0;
}

static int seek_weights(FILE *f, uint64_t offset) {
#ifdef _MSC_VER
    return _fseeki64(f, (int64_t)offset, SEEK_SET);
#else
    return fseeko(f, (off_t)offset, SEEK_SET);
#endif
}

static int tensor_load(PFModel *m, Tensor *t) {
    if (t->data) return 0;
    if (m->weights_map && t->offset <= m->weights_map_size && t->nbytes <= m->weights_map_size - t->offset) {
        t->data = m->weights_map + t->offset;
        return 0;
    }
    t->data = xmalloc((size_t)t->nbytes);
    if (seek_weights(m->weights_file, t->offset) != 0) {
        fprintf(stderr, "seek failed for %s\n", t->name);
        free(t->data);
        t->data = NULL;
        return -1;
    }
    if (fread(t->data, 1, (size_t)t->nbytes, m->weights_file) != (size_t)t->nbytes) {
        fprintf(stderr, "read failed for %s\n", t->name);
        free(t->data);
        t->data = NULL;
        return -1;
    }
    return 0;
}

static void tensor_unload(PFModel *m, Tensor *t) {
    if (!t) return;
    if (t->data && (!m->weights_map || (uint8_t *)t->data < m->weights_map || (uint8_t *)t->data >= m->weights_map + m->weights_map_size)) free(t->data);
    t->data = NULL;
}

static Tensor *load_tensor(PFModel *m, const char *name) {
    Tensor *t = find_tensor(m, name);
    if (!t) return NULL;
    if (tensor_load(m, t) != 0) return NULL;
    return t;
}

static void linear_batch(float *out, const float *x, int rows, int in, int out_dim, Tensor *w, Tensor *b) {
    if (b) {
        for (int r = 0; r < rows; ++r) {
            float *outr = out + (size_t)r * (size_t)out_dim;
            if (b->dtype == 2) {
                memcpy(outr, (const float *)b->data, (size_t)out_dim * sizeof(float));
            } else {
                const uint8_t *bp = (const uint8_t *)b->data;
#ifdef PF_USE_AVX2
                int j = 0;
                for (; j + 7 < out_dim; j += 8) _mm256_storeu_ps(outr + j, pf_load_bf16x8(bp + (size_t)j * 2u));
                for (; j < out_dim; ++j) outr[j] = bf16_to_float(read_le16_mem(bp + (size_t)j * 2u));
#else
                for (int o = 0; o < out_dim; ++o) outr[o] = bf16_to_float(read_le16_mem(bp + (size_t)o * 2u));
#endif
            }
        }
    } else {
        memset(out, 0, (size_t)rows * (size_t)out_dim * sizeof(float));
    }

#ifdef PF_USE_AVX2
    if (w->dtype == 2) {
        const float *wd = (const float *)w->data;
        for (int r = 0; r < rows; ++r) {
            const float *xr = x + (size_t)r * (size_t)in;
            float *outr = out + (size_t)r * (size_t)out_dim;
            int o = 0;
            for (; o + 3 < out_dim; o += 4) {
                const float *w0 = wd + (size_t)(o + 0) * (size_t)in;
                const float *w1 = wd + (size_t)(o + 1) * (size_t)in;
                const float *w2 = wd + (size_t)(o + 2) * (size_t)in;
                const float *w3 = wd + (size_t)(o + 3) * (size_t)in;
                __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                int i = 0;
                for (; i + 7 < in; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xr + i);
                    s0 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w0 + i), s0);
                    s1 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w1 + i), s1);
                    s2 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w2 + i), s2);
                    s3 = _mm256_fmadd_ps(xv, _mm256_loadu_ps(w3 + i), s3);
                }
                outr[o+0] += pf_hsum_f32x8(s0); outr[o+1] += pf_hsum_f32x8(s1);
                outr[o+2] += pf_hsum_f32x8(s2); outr[o+3] += pf_hsum_f32x8(s3);
                for (; i < in; ++i) {
                    float xi = xr[i];
                    outr[o+0] += xi * w0[i]; outr[o+1] += xi * w1[i];
                    outr[o+2] += xi * w2[i]; outr[o+3] += xi * w3[i];
                }
            }
            for (; o < out_dim; ++o) {
                const float *wrow = wd + (size_t)o * (size_t)in;
                __m256 sum = _mm256_setzero_ps();
                int i = 0;
                for (; i + 7 < in; i += 8)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i), _mm256_loadu_ps(wrow + i), sum);
                float dot = pf_hsum_f32x8(sum);
                for (; i < in; ++i) dot += xr[i] * wrow[i];
                outr[o] += dot;
            }
        }
        return;
    }
    {
        const uint8_t *wd = (const uint8_t *)w->data;
        for (int r = 0; r < rows; ++r) {
            const float *xr = x + (size_t)r * (size_t)in;
            float *outr = out + (size_t)r * (size_t)out_dim;
            int o = 0;
            for (; o + 3 < out_dim; o += 4) {
                const uint8_t *w0 = wd + (size_t)(o + 0) * (size_t)in * 2u;
                const uint8_t *w1 = wd + (size_t)(o + 1) * (size_t)in * 2u;
                const uint8_t *w2 = wd + (size_t)(o + 2) * (size_t)in * 2u;
                const uint8_t *w3 = wd + (size_t)(o + 3) * (size_t)in * 2u;
                __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                int i = 0;
                for (; i + 7 < in; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xr + i);
                    s0 = _mm256_fmadd_ps(xv, pf_load_bf16x8(w0 + (size_t)i * 2u), s0);
                    s1 = _mm256_fmadd_ps(xv, pf_load_bf16x8(w1 + (size_t)i * 2u), s1);
                    s2 = _mm256_fmadd_ps(xv, pf_load_bf16x8(w2 + (size_t)i * 2u), s2);
                    s3 = _mm256_fmadd_ps(xv, pf_load_bf16x8(w3 + (size_t)i * 2u), s3);
                }
                outr[o+0] += pf_hsum_f32x8(s0); outr[o+1] += pf_hsum_f32x8(s1);
                outr[o+2] += pf_hsum_f32x8(s2); outr[o+3] += pf_hsum_f32x8(s3);
                for (; i < in; ++i) {
                    float xi = xr[i];
                    outr[o+0] += xi * bf16_to_float(read_le16_mem(w0 + (size_t)i * 2u));
                    outr[o+1] += xi * bf16_to_float(read_le16_mem(w1 + (size_t)i * 2u));
                    outr[o+2] += xi * bf16_to_float(read_le16_mem(w2 + (size_t)i * 2u));
                    outr[o+3] += xi * bf16_to_float(read_le16_mem(w3 + (size_t)i * 2u));
                }
            }
            for (; o < out_dim; ++o) {
                const uint8_t *wrow = wd + (size_t)o * (size_t)in * 2u;
                __m256 sum = _mm256_setzero_ps();
                int i = 0;
                for (; i + 7 < in; i += 8)
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i), pf_load_bf16x8(wrow + (size_t)i * 2u), sum);
                float dot = pf_hsum_f32x8(sum);
                for (; i < in; ++i) dot += xr[i] * bf16_to_float(read_le16_mem(wrow + (size_t)i * 2u));
                outr[o] += dot;
            }
        }
        return;
    }
#else
    if (w->dtype == 2) {
        const float *wd = (const float *)w->data;
        for (int o = 0; o < out_dim; ++o) {
            const float *wrow = wd + (size_t)o * (size_t)in;
            for (int i = 0; i < in; ++i) {
                float wv = wrow[i];
                for (int r = 0; r < rows; ++r) {
                    out[(size_t)r * (size_t)out_dim + (size_t)o] += x[(size_t)r * (size_t)in + (size_t)i] * wv;
                }
            }
        }
        return;
    }

    {
        const uint8_t *wd = (const uint8_t *)w->data;
        for (int o = 0; o < out_dim; ++o) {
            const uint8_t *wrow = wd + (size_t)o * (size_t)in * 2u;
            for (int i = 0; i < in; ++i) {
                float wv = bf16_to_float(read_le16_mem(wrow + (size_t)i * 2u));
                for (int r = 0; r < rows; ++r) {
                    out[(size_t)r * (size_t)out_dim + (size_t)o] += x[(size_t)r * (size_t)in + (size_t)i] * wv;
                }
            }
        }
    }
#endif
}

static void matvec_transposed(float *out, const float *x, int in, int out_dim, Tensor *w, Tensor *b, int expert) {
    if (b) {
        if (b->dtype == 2) {
            memcpy(out, (const float *)b->data + (size_t)expert * (size_t)out_dim, (size_t)out_dim * sizeof(float));
        } else {
            const uint8_t *bp = (const uint8_t *)b->data + (uint64_t)expert * (uint64_t)out_dim * 2u;
#ifdef PF_USE_AVX2
            int o = 0;
            for (; o + 7 < out_dim; o += 8) _mm256_storeu_ps(out + o, pf_load_bf16x8(bp + (size_t)o * 2u));
            for (; o < out_dim; ++o) out[o] = bf16_to_float(read_le16_mem(bp + (size_t)o * 2u));
#else
            for (int o = 0; o < out_dim; ++o) out[o] = bf16_to_float(read_le16_mem(bp + (size_t)o * 2u));
#endif
        }
    } else {
        memset(out, 0, (size_t)out_dim * sizeof(float));
    }

    uint64_t expert_base = (uint64_t)expert * (uint64_t)in * (uint64_t)out_dim;
#ifdef PF_USE_AVX2
    if (w->dtype == 2) {
        const float *wd = (const float *)w->data + expert_base;
        for (int i = 0; i < in; ++i) {
            __m256 xi = _mm256_set1_ps(x[i]);
            const float *row = wd + (size_t)i * (size_t)out_dim;
            int o = 0;
            for (; o + 7 < out_dim; o += 8) {
                __m256 acc = _mm256_loadu_ps(out + o);
                _mm256_storeu_ps(out + o, _mm256_fmadd_ps(xi, _mm256_loadu_ps(row + o), acc));
            }
            for (; o < out_dim; ++o) out[o] += x[i] * row[o];
        }
        return;
    }
    {
        const uint8_t *wd = (const uint8_t *)w->data + expert_base * 2u;
        for (int i = 0; i < in; ++i) {
            __m256 xi = _mm256_set1_ps(x[i]);
            const uint8_t *row = wd + (size_t)i * (size_t)out_dim * 2u;
            int o = 0;
            for (; o + 7 < out_dim; o += 8) {
                __m256 acc = _mm256_loadu_ps(out + o);
                _mm256_storeu_ps(out + o, _mm256_fmadd_ps(xi, pf_load_bf16x8(row + (size_t)o * 2u), acc));
            }
            for (; o < out_dim; ++o) out[o] += x[i] * bf16_to_float(read_le16_mem(row + (size_t)o * 2u));
        }
        return;
    }
#else
    if (w->dtype == 2) {
        const float *wd = (const float *)w->data + expert_base;
        for (int i = 0; i < in; ++i) {
            float xi = x[i];
            const float *row = wd + (size_t)i * (size_t)out_dim;
            int o = 0;
            for (; o + 3 < out_dim; o += 4) {
                out[o] += xi * row[o];
                out[o + 1] += xi * row[o + 1];
                out[o + 2] += xi * row[o + 2];
                out[o + 3] += xi * row[o + 3];
            }
            for (; o < out_dim; ++o) out[o] += xi * row[o];
        }
        return;
    }

    {
        const uint8_t *wd = (const uint8_t *)w->data + expert_base * 2u;
        for (int i = 0; i < in; ++i) {
            float xi = x[i];
            const uint8_t *row = wd + (size_t)i * (size_t)out_dim * 2u;
            int o = 0;
            for (; o + 3 < out_dim; o += 4) {
                out[o] += xi * bf16_to_float(read_le16_mem(row + (size_t)o * 2u));
                out[o + 1] += xi * bf16_to_float(read_le16_mem(row + (size_t)(o + 1) * 2u));
                out[o + 2] += xi * bf16_to_float(read_le16_mem(row + (size_t)(o + 2) * 2u));
                out[o + 3] += xi * bf16_to_float(read_le16_mem(row + (size_t)(o + 3) * 2u));
            }
            for (; o < out_dim; ++o) out[o] += xi * bf16_to_float(read_le16_mem(row + (size_t)o * 2u));
        }
    }
#endif
}

static void rms_norm(float *out, const float *x, Tensor *weight, int n) {
#ifdef PF_USE_AVX2
    __m256 ss_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        ss_vec = _mm256_fmadd_ps(xv, xv, ss_vec);
    }
    float ss = pf_hsum_f32x8(ss_vec);
    for (; i < n; ++i) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + PF_RMS_EPS);
    __m256 sv = _mm256_set1_ps(scale);
    if (weight->dtype == 2) {
        const float *w = (const float *)weight->data;
        for (i = 0; i + 7 < n; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 wv = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, sv), wv));
        }
        for (; i < n; ++i) out[i] = x[i] * scale * w[i];
        return;
    }
    const uint8_t *w = (const uint8_t *)weight->data;
    for (i = 0; i + 7 < n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = pf_load_bf16x8(w + (size_t)i * 2u);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(xv, sv), wv));
    }
    for (; i < n; ++i) out[i] = x[i] * scale * bf16_to_float(read_le16_mem(w + (size_t)i * 2u));
#else
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * (double)x[i];
    float scale = 1.0f / sqrtf((float)(ss / n) + PF_RMS_EPS);
    if (weight->dtype == 2) {
        const float *w = (const float *)weight->data;
        for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
        return;
    }
    const uint8_t *w = (const uint8_t *)weight->data;
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * bf16_to_float(read_le16_mem(w + (size_t)i * 2u));
#endif
}

static float sigmoidf_fast(float x) {
    if (x < -40.0f) return 0.0f;
    if (x > 40.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void topk4(const float *values, int n, int *idx, float *val) {
    for (int k = 0; k < PF_TOPK; ++k) {
        idx[k] = -1;
        val[k] = -FLT_MAX;
    }
    for (int i = 0; i < n; ++i) {
        float v = values[i];
        for (int k = 0; k < PF_TOPK; ++k) {
            if (v > val[k]) {
                for (int j = PF_TOPK - 1; j > k; --j) {
                    val[j] = val[j - 1];
                    idx[j] = idx[j - 1];
                }
                val[k] = v;
                idx[k] = i;
                break;
            }
        }
    }
}

static void softmax_inplace(float *v, int n) {
    float maxv = v[0];
    for (int i = 1; i < n; ++i) if (v[i] > maxv) maxv = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        v[i] = expf(v[i] - maxv);
        sum += v[i];
    }
    if (sum == 0.0f) sum = 1.0f;
    for (int i = 0; i < n; ++i) v[i] /= sum;
}

static void rope_freq(int pos, int half_dim, float *cos_out, float *sin_out) {
    float d_half = PF_HEAD_DIM / 2.0f;
    float low = d_half * logf(PF_ROPE_ORIGINAL_MAX / (PF_ROPE_BETA_FAST * 2.0f * (float)M_PI)) / logf(PF_ROPE_THETA);
    float high = d_half * logf(PF_ROPE_ORIGINAL_MAX / (PF_ROPE_BETA_SLOW * 2.0f * (float)M_PI)) / logf(PF_ROPE_THETA);
    float concentration = 0.1f * logf(PF_ROPE_FACTOR) + 1.0f;
    for (int i = 0; i < half_dim; ++i) {
        float freq = powf(PF_ROPE_THETA, (float)(2 * i) / (float)PF_HEAD_DIM);
        float interpolation = 1.0f / (PF_ROPE_FACTOR * freq);
        float extrapolation = 1.0f / freq;
        float ramp = ((float)i - low) / (high - low);
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        float mask = 1.0f - ramp;
        float inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
        float angle = (float)pos * inv_freq;
        cos_out[i] = cosf(angle) * concentration;
        sin_out[i] = sinf(angle) * concentration;
    }
}

static int ensure_rope_cache(PFModel *m, int ntok) {
    if (ntok <= m->rope_tokens) return 0;
    size_t n = (size_t)ntok * (PF_HEAD_DIM / 2);
    if (grow_float_buffer(&m->rope_cos, n) != 0) return -1;
    if (grow_float_buffer(&m->rope_sin, n) != 0) return -1;
    for (int pos = m->rope_tokens; pos < ntok; ++pos) {
        rope_freq(pos, PF_HEAD_DIM / 2, m->rope_cos + (size_t)pos * (PF_HEAD_DIM / 2), m->rope_sin + (size_t)pos * (PF_HEAD_DIM / 2));
    }
    m->rope_tokens = ntok;
    return 0;
}

static void apply_rope_cached(float *x, const float *c, const float *s) {
    for (int i = 0; i < PF_HEAD_DIM / 2; ++i) {
        float a = x[2 * i];
        float b = x[2 * i + 1];
        x[2 * i] = a * c[i] - b * s[i];
        x[2 * i + 1] = b * c[i] + a * s[i];
    }
}

static void unload_layer_tensors(PFModel *m, Tensor **ts, int n) {
    for (int i = 0; i < n; ++i) tensor_unload(m, ts[i]);
}

static int load_layer_tensors(PFModel *m, int layer, Tensor **in_norm, Tensor **post_norm, Tensor **q_w, Tensor **q_b, Tensor **k_w, Tensor **k_b, Tensor **v_w, Tensor **v_b, Tensor **o_w, Tensor **o_b, Tensor **sinks, Tensor **router_w, Tensor **router_b, Tensor **up_w, Tensor **up_b, Tensor **down_w, Tensor **down_b) {
    char name[160];
    Tensor *loaded[17];
    int loaded_count = 0;
#define LT(dst, fmt) do { \
        snprintf(name, sizeof(name), fmt, layer); \
        *(dst) = load_tensor(m, name); \
        if (!*(dst)) { unload_layer_tensors(m, loaded, loaded_count); return -1; } \
        loaded[loaded_count++] = *(dst); \
    } while (0)
    LT(in_norm, "model.layers.%d.input_layernorm.weight");
    LT(post_norm, "model.layers.%d.post_attention_layernorm.weight");
    LT(q_w, "model.layers.%d.self_attn.q_proj.weight");
    LT(q_b, "model.layers.%d.self_attn.q_proj.bias");
    LT(k_w, "model.layers.%d.self_attn.k_proj.weight");
    LT(k_b, "model.layers.%d.self_attn.k_proj.bias");
    LT(v_w, "model.layers.%d.self_attn.v_proj.weight");
    LT(v_b, "model.layers.%d.self_attn.v_proj.bias");
    LT(o_w, "model.layers.%d.self_attn.o_proj.weight");
    LT(o_b, "model.layers.%d.self_attn.o_proj.bias");
    LT(sinks, "model.layers.%d.self_attn.sinks");
    LT(router_w, "model.layers.%d.mlp.router.weight");
    LT(router_b, "model.layers.%d.mlp.router.bias");
    LT(up_w, "model.layers.%d.mlp.experts.gate_up_proj");
    LT(up_b, "model.layers.%d.mlp.experts.gate_up_proj_bias");
    LT(down_w, "model.layers.%d.mlp.experts.down_proj");
    LT(down_b, "model.layers.%d.mlp.experts.down_proj_bias");
#undef LT
    return 0;
}

static int ensure_buffers(PFModel *m, int ntok) {
    if (ntok <= m->buffer_tokens) return ensure_rope_cache(m, ntok);
    size_t n = (size_t)ntok, cnt;
    if (checked_mul_size(n, PF_HIDDEN, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->x, cnt) != 0) return -1;
    if (grow_float_buffer(&m->norm, cnt) != 0) return -1;
    if (checked_mul3_size(n, PF_HEADS, PF_HEAD_DIM, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->q, cnt) != 0) return -1;
    if (grow_float_buffer(&m->attn, cnt) != 0) return -1;
    if (checked_mul3_size(n, PF_KV_HEADS, PF_HEAD_DIM, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->k, cnt) != 0) return -1;
    if (grow_float_buffer(&m->v, cnt) != 0) return -1;
    if (checked_mul_size(n, PF_HIDDEN, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->moe, cnt) != 0) return -1;
    if (checked_mul_size(n, PF_EXPERTS, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->router_logits, cnt) != 0) return -1;
    if (checked_mul_size(n, PF_LABELS, &cnt) != 0) return -1;
    if (grow_float_buffer(&m->logits, cnt) != 0) return -1;
    if (ensure_rope_cache(m, ntok) != 0) return -1;
    m->buffer_tokens = ntok;
    return 0;
}

static int ensure_moe_buffers(PFModel *m, int ntok, int nthreads) {
    size_t assignments;
    if (checked_mul_size((size_t)ntok, PF_TOPK, &assignments) != 0 || assignments > INT_MAX) return -1;
    if ((int)assignments > m->moe_assignments) {
        if (grow_int_buffer(&m->moe_expert, assignments) != 0) return -1;
        if (grow_int_buffer(&m->moe_token, assignments) != 0) return -1;
        if (grow_int_buffer(&m->moe_sorted_token, assignments) != 0) return -1;
        if (grow_float_buffer(&m->moe_weight, assignments) != 0) return -1;
        if (grow_float_buffer(&m->moe_sorted_weight, assignments) != 0) return -1;
        m->moe_assignments = (int)assignments;
    }
    size_t scratch_per_thread;
    size_t scratch_count;
    if (checked_mul_size((size_t)ntok, PF_HIDDEN, &scratch_per_thread) != 0) return -1;
    if (checked_mul_size((size_t)nthreads, scratch_per_thread, &scratch_count) != 0) return -1;
    if (scratch_count > m->moe_thread_scratch_count) {
        if (grow_float_buffer(&m->moe_thread_scratch, scratch_count) != 0) return -1;
        m->moe_thread_scratch_count = scratch_count;
    }
    m->moe_scratch_threads = nthreads;
    return 0;
}

static int forward_tokens(PFModel *m, const PFToken *tokens, int ntok, float *logits_out) {
    Tensor *embed = load_tensor(m, "model.embed_tokens.weight");
    if (!embed) return -1;
    for (int t = 0; t < ntok; ++t) {
        int id = tokens[t].id;
        if (id < 0 || id >= PF_VOCAB_SIZE) id = PF_PAD_TOKEN;
        copy_tensor_row(m->x + t * PF_HIDDEN, embed, (uint64_t)id, PF_HIDDEN);
    }
        tensor_unload(m, embed);

    const float qk_scale = powf((float)PF_HEAD_DIM, -0.25f);
    const int q_per_kv = PF_HEADS / PF_KV_HEADS;

    for (int layer = 0; layer < PF_LAYERS; ++layer) {
        Tensor *in_norm, *post_norm, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b, *sinks, *router_w, *router_b, *up_w, *up_b, *down_w, *down_b;
        Tensor *loaded[17];
        if (load_layer_tensors(m, layer, &in_norm, &post_norm, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b, &sinks, &router_w, &router_b, &up_w, &up_b, &down_w, &down_b) != 0) return -1;
        loaded[0]=in_norm; loaded[1]=post_norm; loaded[2]=q_w; loaded[3]=q_b; loaded[4]=k_w; loaded[5]=k_b; loaded[6]=v_w; loaded[7]=v_b; loaded[8]=o_w; loaded[9]=o_b; loaded[10]=sinks; loaded[11]=router_w; loaded[12]=router_b; loaded[13]=up_w; loaded[14]=up_b; loaded[15]=down_w; loaded[16]=down_b;

        for (int t = 0; t < ntok; ++t) rms_norm(m->norm + t * PF_HIDDEN, m->x + t * PF_HIDDEN, in_norm, PF_HIDDEN);
        linear_batch(m->q, m->norm, ntok, PF_HIDDEN, PF_HEADS * PF_HEAD_DIM, q_w, q_b);
        linear_batch(m->k, m->norm, ntok, PF_HIDDEN, PF_KV_HEADS * PF_HEAD_DIM, k_w, k_b);
        linear_batch(m->v, m->norm, ntok, PF_HIDDEN, PF_KV_HEADS * PF_HEAD_DIM, v_w, v_b);
        for (int t = 0; t < ntok; ++t) {
            const float *rope_c = m->rope_cos + (size_t)t * (PF_HEAD_DIM / 2);
            const float *rope_s = m->rope_sin + (size_t)t * (PF_HEAD_DIM / 2);
            for (int h = 0; h < PF_HEADS; ++h) {
                float *qh = m->q + t * PF_HEADS * PF_HEAD_DIM + h * PF_HEAD_DIM;
                apply_rope_cached(qh, rope_c, rope_s);
                for (int d = 0; d < PF_HEAD_DIM; ++d) qh[d] *= qk_scale;
            }
            for (int h = 0; h < PF_KV_HEADS; ++h) {
                float *kh = m->k + t * PF_KV_HEADS * PF_HEAD_DIM + h * PF_HEAD_DIM;
                apply_rope_cached(kh, rope_c, rope_s);
                for (int d = 0; d < PF_HEAD_DIM; ++d) kh[d] *= qk_scale;
            }
        }
        memset(m->attn, 0, (size_t)ntok * PF_HEADS * PF_HEAD_DIM * sizeof(float));
        for (int t = 0; t < ntok; ++t) {
            int start = t - PF_SLIDING_WINDOW;
            if (start < 0) start = 0;
            int end = t + PF_SLIDING_WINDOW;
            if (end >= ntok) end = ntok - 1;
            int win = end - start + 1;
            for (int h = 0; h < PF_HEADS; ++h) {
                int kvh = h / q_per_kv;
                float scores[PF_SLIDING_WINDOW * 2 + 1];
                float maxv = -FLT_MAX;
                float *qh = m->q + t * PF_HEADS * PF_HEAD_DIM + h * PF_HEAD_DIM;
                for (int j = start; j <= end; ++j) {
                    float *kh = m->k + j * PF_KV_HEADS * PF_HEAD_DIM + kvh * PF_HEAD_DIM;
#ifdef PF_USE_AVX2
                    __m256 dv = _mm256_setzero_ps();
                    for (int d = 0; d < PF_HEAD_DIM; d += 8)
                        dv = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), _mm256_loadu_ps(kh + d), dv);
                    float dot = pf_hsum_f32x8(dv);
#else
                    float dot = 0.0f;
                    for (int d = 0; d < PF_HEAD_DIM; ++d) dot += qh[d] * kh[d];
#endif
                    scores[j - start] = dot;
                    if (dot > maxv) maxv = dot;
                }
                float sink = tensor_get(sinks, h);
                if (sink > maxv) maxv = sink;
                float sum = expf(sink - maxv);
                for (int i = 0; i < win; ++i) {
                    scores[i] = expf(scores[i] - maxv);
                    sum += scores[i];
                }
                float inv_sum = 1.0f / sum;
                float *aout = m->attn + t * PF_HEADS * PF_HEAD_DIM + h * PF_HEAD_DIM;
                for (int j = start; j <= end; ++j) {
                    float aw = scores[j - start] * inv_sum;
                    float *vv = m->v + j * PF_KV_HEADS * PF_HEAD_DIM + kvh * PF_HEAD_DIM;
#ifdef PF_USE_AVX2
                    __m256 wv = _mm256_set1_ps(aw);
                    for (int d = 0; d < PF_HEAD_DIM; d += 8) {
                        __m256 ov = _mm256_loadu_ps(aout + d);
                        _mm256_storeu_ps(aout + d, _mm256_fmadd_ps(wv, _mm256_loadu_ps(vv + d), ov));
                    }
#else
                    for (int d = 0; d < PF_HEAD_DIM; ++d) aout[d] += aw * vv[d];
#endif
                }
            }
        }
        linear_batch(m->moe, m->attn, ntok, PF_HEADS * PF_HEAD_DIM, PF_HIDDEN, o_w, o_b);
        for (int t = 0; t < ntok; ++t) {
            for (int i = 0; i < PF_HIDDEN; ++i) m->x[t * PF_HIDDEN + i] += m->moe[t * PF_HIDDEN + i];
        }

        for (int t = 0; t < ntok; ++t) rms_norm(m->norm + t * PF_HIDDEN, m->x + t * PF_HIDDEN, post_norm, PF_HIDDEN);
        linear_batch(m->router_logits, m->norm, ntok, PF_HIDDEN, PF_EXPERTS, router_w, router_b);
        memset(m->moe, 0, (size_t)ntok * PF_HIDDEN * sizeof(float));
        {
            int nassign = ntok * PF_TOPK;
            int nthreads = 1;
#ifdef _OPENMP
            #pragma omp parallel
            { if (omp_get_thread_num() == 0) nthreads = omp_get_num_threads(); }
#endif
            if (ensure_moe_buffers(m, ntok, nthreads) != 0) {
                unload_layer_tensors(m, loaded, 17);
                return -1;
            }
            int *ae = m->moe_expert;
            int *at = m->moe_token;
            float *aw = m->moe_weight;
            int *st = m->moe_sorted_token;
            float *sw = m->moe_sorted_weight;
            int ai = 0;
            for (int t = 0; t < ntok; ++t) {
                int top_idx[PF_TOPK];
                float top_val[PF_TOPK];
                topk4(m->router_logits + t * PF_EXPERTS, PF_EXPERTS, top_idx, top_val);
                softmax_inplace(top_val, PF_TOPK);
                for (int kk = 0; kk < PF_TOPK; ++kk) {
                    ae[ai] = top_idx[kk];
                    at[ai] = t;
                    aw[ai] = top_val[kk];
                    ai++;
                }
            }
            int ec[PF_EXPERTS], eo[PF_EXPERTS + 1], ep[PF_EXPERTS];
            memset(ec, 0, sizeof(ec));
            for (int a = 0; a < nassign; ++a) ec[ae[a]]++;
            eo[0] = 0;
            for (int e = 0; e < PF_EXPERTS; ++e) eo[e + 1] = eo[e] + ec[e];
            memcpy(ep, eo, sizeof(ep));
            for (int a = 0; a < nassign; ++a) {
                int p = ep[ae[a]]++;
                st[p] = at[a];
                sw[p] = aw[a];
            }
            size_t scratch_count = (size_t)nthreads * (size_t)ntok * PF_HIDDEN;
            memset(m->moe_thread_scratch, 0, scratch_count * sizeof(float));
            {
            int e;
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
#endif
            for (e = 0; e < PF_EXPERTS; ++e) {
                if (ec[e] == 0) continue;
                float lgup[2 * PF_INTERMEDIATE];
                float lehid[PF_INTERMEDIATE];
                float lres[PF_HIDDEN];
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float *lmoe = m->moe_thread_scratch + (size_t)tid * (size_t)ntok * PF_HIDDEN;
                for (int a = eo[e]; a < eo[e + 1]; ++a) {
                    int tt = st[a];
                    matvec_transposed(lgup, m->norm + tt * PF_HIDDEN, PF_HIDDEN, 2 * PF_INTERMEDIATE, up_w, up_b, e);
                    for (int i = 0; i < PF_INTERMEDIATE; ++i) {
                        float gate = lgup[i];
                        float up = lgup[PF_INTERMEDIATE + i];
                        if (gate > PF_SWIGLU_LIMIT) gate = PF_SWIGLU_LIMIT;
                        if (up > PF_SWIGLU_LIMIT) up = PF_SWIGLU_LIMIT;
                        if (up < -PF_SWIGLU_LIMIT) up = -PF_SWIGLU_LIMIT;
                        float glu = gate * sigmoidf_fast(gate * PF_SWIGLU_ALPHA);
                        lehid[i] = (up + 1.0f) * glu;
                    }
                    matvec_transposed(lres, lehid, PF_INTERMEDIATE, PF_HIDDEN, down_w, down_b, e);
                    float wt = sw[a];
#ifdef PF_USE_AVX2
                    __m256 wv = _mm256_set1_ps(wt);
                    for (int i = 0; i < PF_HIDDEN; i += 8) {
                        __m256 mv = _mm256_loadu_ps(lmoe + tt * PF_HIDDEN + i);
                        __m256 rv = _mm256_loadu_ps(lres + i);
                        _mm256_storeu_ps(lmoe + tt * PF_HIDDEN + i, _mm256_fmadd_ps(wv, rv, mv));
                    }
#else
                    for (int i = 0; i < PF_HIDDEN; ++i) lmoe[tt * PF_HIDDEN + i] += lres[i] * wt;
#endif
                }
            }
            }
            for (int tid = 0; tid < nthreads; ++tid) {
                const float *src = m->moe_thread_scratch + (size_t)tid * (size_t)ntok * PF_HIDDEN;
                for (int i = 0; i < ntok * PF_HIDDEN; ++i) m->moe[i] += src[i];
            }
        }
        for (int t = 0; t < ntok; ++t) {
            for (int i = 0; i < PF_HIDDEN; ++i) m->x[t * PF_HIDDEN + i] += m->moe[t * PF_HIDDEN + i];
        }
        unload_layer_tensors(m, loaded, 17);
    }

    Tensor *final_norm = load_tensor(m, "model.norm.weight");
    Tensor *score_w = load_tensor(m, "score.weight");
    Tensor *score_b = find_tensor_optional(m, "score.bias");
    if (score_b && tensor_load(m, score_b) != 0) score_b = NULL;
    if (!final_norm || !score_w) {
        tensor_unload(m, final_norm);
        tensor_unload(m, score_w);
        tensor_unload(m, score_b);
        return -1;
    }
    for (int t = 0; t < ntok; ++t) {
        rms_norm(m->norm + t * PF_HIDDEN, m->x + t * PF_HIDDEN, final_norm, PF_HIDDEN);
    }
    linear_batch(logits_out, m->norm, ntok, PF_HIDDEN, PF_LABELS, score_w, score_b);
    tensor_unload(m, final_norm);
    tensor_unload(m, score_w);
    if (score_b) tensor_unload(m, score_b);
    return 0;
}

static int match_special(PFModel *m, const uint8_t *s, uint32_t pos, uint32_t len, int *id, uint32_t *match_len) {
    for (int i = 0; i < m->special_count; ++i) {
        VocabEntry *e = &m->specials[i];
        if (pos + e->len <= len && memcmp(s + pos, e->bytes, e->len) == 0) {
            *id = e->id;
            *match_len = e->len;
            return 1;
        }
    }
    return 0;
}

int pf_tokenize(PFModel *model, const char *text, PFToken **tokens_out, int *count_out) {
    if (!model || !text || !tokens_out || !count_out) return -1;
    size_t text_len = strlen(text);
    if (text_len > UINT32_MAX) return -1;
    const uint8_t *s = (const uint8_t *)text;
    uint32_t len = (uint32_t)text_len;
    uint32_t cap = 128;
    uint32_t count = 0;
    PFToken *tokens = (PFToken *)xmalloc(cap * sizeof(PFToken));
    uint32_t pos = 0;
    while (pos < len) {
        int sid;
        uint32_t slen;
        if (match_special(model, s, pos, len, &sid, &slen)) {
            if (count == cap) { cap *= 2; tokens = (PFToken *)xreallocarray(tokens, cap, sizeof(PFToken)); }
            tokens[count++] = (PFToken){sid, pos, pos + slen};
            pos += slen;
            continue;
        }
        TrieNode *cur = model->trie;
        int best_id = -1;
        uint32_t best_end = pos;
        uint32_t p = pos;
        while (p < len && cur->child[s[p]]) {
            cur = cur->child[s[p]];
            ++p;
            if (cur->token_id >= 0) {
                best_id = cur->token_id;
                best_end = p;
            }
        }
        if (best_id < 0) {
            fprintf(stderr, "tokenizer failed at byte offset %u\n", pos);
            free(tokens);
            return -1;
        }
        if (count == cap) {
            cap *= 2;
            tokens = (PFToken *)xreallocarray(tokens, cap, sizeof(PFToken));
        }
        tokens[count++] = (PFToken){best_id, pos, best_end};
        pos = best_end;
    }
    *tokens_out = tokens;
    *count_out = (int)count;
    return 0;
}

static int valid_transition(int prev, int next) {
    int prev_span = PF_TOKEN_TO_SPAN[prev];
    int next_span = PF_TOKEN_TO_SPAN[next];
    char prev_tag = PF_TOKEN_BOUNDARY[prev];
    char next_tag = PF_TOKEN_BOUNDARY[next];
    int prev_bg = prev_span == 0 || prev == 0;
    int next_bg = next_span == 0 || next == 0;
    if (prev_bg) return next_bg || next_tag == 'B' || next_tag == 'S';
    if (prev_tag == 'B' || prev_tag == 'I') return (next_tag == 'I' || next_tag == 'E') && prev_span == next_span;
    if (prev_tag == 'E' || prev_tag == 'S') return next_bg || next_tag == 'B' || next_tag == 'S';
    return next_bg;
}

static float transition_bias(PFModel *m, int prev, int next) {
    int prev_span = PF_TOKEN_TO_SPAN[prev];
    int next_span = PF_TOKEN_TO_SPAN[next];
    char prev_tag = PF_TOKEN_BOUNDARY[prev];
    char next_tag = PF_TOKEN_BOUNDARY[next];
    int prev_bg = prev_span == 0 || prev == 0;
    int next_bg = next_span == 0 || next == 0;
    if (prev_bg) {
        if (next_bg) return m->transition_bias_background_stay;
        if (next_tag == 'B' || next_tag == 'S') return m->transition_bias_background_to_start;
        return 0.0f;
    }
    if (prev_tag == 'B' || prev_tag == 'I') {
        if (next_tag == 'I' && prev_span == next_span) return m->transition_bias_inside_to_continue;
        if (next_tag == 'E' && prev_span == next_span) return m->transition_bias_inside_to_end;
        return 0.0f;
    }
    if (prev_tag == 'E' || prev_tag == 'S') {
        if (next_bg) return m->transition_bias_end_to_background;
        if (next_tag == 'B' || next_tag == 'S') return m->transition_bias_end_to_start;
        return 0.0f;
    }
    return 0.0f;
}

static void init_viterbi(PFModel *m) {
    m->transition = (float *)xmalloc(PF_LABELS * PF_LABELS * sizeof(float));
    m->start_scores = (float *)xmalloc(PF_LABELS * sizeof(float));
    m->end_scores = (float *)xmalloc(PF_LABELS * sizeof(float));
    for (int i = 0; i < PF_LABELS; ++i) {
        char tag = PF_TOKEN_BOUNDARY[i];
        m->start_scores[i] = (tag == 'B' || tag == 'S' || i == 0) ? 0.0f : PF_NEG_INF;
        m->end_scores[i] = (tag == 'E' || tag == 'S' || i == 0) ? 0.0f : PF_NEG_INF;
        for (int j = 0; j < PF_LABELS; ++j) m->transition[i * PF_LABELS + j] = valid_transition(i, j) ? transition_bias(m, i, j) : PF_NEG_INF;
    }
}

static int *viterbi_decode(PFModel *m, float *logits, int ntok) {
    float *dp = (float *)xmalloc((size_t)ntok * PF_LABELS * sizeof(float));
    int *back = (int *)xmalloc((size_t)ntok * PF_LABELS * sizeof(int));
    int *labels = (int *)xmalloc((size_t)ntok * sizeof(int));
    for (int c = 0; c < PF_LABELS; ++c) {
        dp[c] = m->start_scores[c] + logits[c];
        back[c] = -1;
    }
    for (int t = 1; t < ntok; ++t) {
        for (int c = 0; c < PF_LABELS; ++c) {
            float best = PF_NEG_INF;
            int best_p = 0;
            for (int p = 0; p < PF_LABELS; ++p) {
                float score = dp[(t - 1) * PF_LABELS + p] + m->transition[p * PF_LABELS + c];
                if (score > best) {
                    best = score;
                    best_p = p;
                }
            }
            dp[t * PF_LABELS + c] = best + logits[t * PF_LABELS + c];
            back[t * PF_LABELS + c] = best_p;
        }
    }
    float best = PF_NEG_INF;
    int cur = 0;
    for (int c = 0; c < PF_LABELS; ++c) {
        float score = dp[(ntok - 1) * PF_LABELS + c] + m->end_scores[c];
        if (score > best) {
            best = score;
            cur = c;
        }
    }
    for (int t = ntok - 1; t >= 0; --t) {
        labels[t] = cur;
        cur = back[t * PF_LABELS + cur];
        if (cur < 0) cur = 0;
    }
    free(dp);
    free(back);
    return labels;
}

static void copy_label(char dst[32], const char *src) {
#ifdef _MSC_VER
    strncpy_s(dst, 32, src, _TRUNCATE);
#else
    snprintf(dst, 32, "%s", src);
#endif
}

static void emit_span(PFSpan **spans, int *cap, int *count, int label, uint32_t start, uint32_t end) {
    if (*count == *cap) {
        *cap *= 2;
        *spans = (PFSpan *)xreallocarray(*spans, (size_t)(*cap), sizeof(PFSpan));
    }
    copy_label((*spans)[*count].label, PF_SPAN_NAMES[label]);
    (*spans)[*count].start = start;
    (*spans)[*count].end = end;
    ++(*count);
}

static int labels_to_spans(const PFToken *tokens, const int *labels, int ntok, PFSpan **spans_out, int *count_out) {
    int cap = 16, count = 0;
    PFSpan *spans = (PFSpan *)xmalloc((size_t)cap * sizeof(PFSpan));
    int current = -1;
    int start_tok = -1;
    for (int i = 0; i < ntok; ++i) {
        int label = labels[i];
        int span = PF_TOKEN_TO_SPAN[label];
        char tag = PF_TOKEN_BOUNDARY[label];
        if (span == 0) {
            if (current > 0 && start_tok >= 0) emit_span(&spans, &cap, &count, current, tokens[start_tok].start, tokens[i - 1].end);
            current = -1;
            start_tok = -1;
            continue;
        }
        if (tag == 'S') {
            if (current > 0 && start_tok >= 0) emit_span(&spans, &cap, &count, current, tokens[start_tok].start, tokens[i - 1].end);
            emit_span(&spans, &cap, &count, span, tokens[i].start, tokens[i].end);
            current = -1;
            start_tok = -1;
        } else if (tag == 'B') {
            if (current > 0 && start_tok >= 0) emit_span(&spans, &cap, &count, current, tokens[start_tok].start, tokens[i - 1].end);
            current = span;
            start_tok = i;
        } else if (tag == 'I') {
            if (current != span) {
                current = span;
                start_tok = i;
            }
        } else if (tag == 'E') {
            if (current != span) start_tok = i;
            emit_span(&spans, &cap, &count, span, tokens[start_tok].start, tokens[i].end);
            current = -1;
            start_tok = -1;
        }
    }
    if (current > 0 && start_tok >= 0) emit_span(&spans, &cap, &count, current, tokens[start_tok].start, tokens[ntok - 1].end);
    *spans_out = spans;
    *count_out = count;
    return 0;
}

static int byte_is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static int is_alnum(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }
static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
static int is_alnum_dot_dash(unsigned char c) { return is_alnum(c) || c == '.' || c == '-' || c == '_'; }

static int is_url_body_char(unsigned char c) {
    return is_alnum(c) || c == '-' || c == '.' || c == '_' || c == '/' ||
           c == '?' || c == '&' || c == '=' || c == '#' || c == '%' ||
           c == '+' || c == ':' || c == '@' || c == '~' || c == '!' ||
           c == '(' || c == ')' || c == ',' || c == ';';
}

static int try_match_intl_phone(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    if (text[pos] != '+') return 0;
    uint32_t p = pos + 1;
    int cc_digits = 0;
    while (p < len && is_digit((unsigned char)text[p]) && cc_digits < 3) { cc_digits++; p++; }
    if (cc_digits < 1) return 0;
    int total_digits = cc_digits, groups = 1;
    while (p < len) {
        unsigned char ch = (unsigned char)text[p];
        if (ch == ' ' || ch == '-' || ch == '.') { p++; }
        else if (ch == '(') {
            p++;
            while (p < len && is_digit((unsigned char)text[p])) { total_digits++; p++; }
            if (p < len && text[p] == ')') p++;
            groups++;
        } else if (is_digit(ch)) {
            while (p < len && is_digit((unsigned char)text[p])) { total_digits++; p++; }
            groups++;
        } else { break; }
    }
    while (p > pos + 1 + (uint32_t)cc_digits && !is_digit((unsigned char)text[p - 1]) && text[p - 1] != ')') p--;
    if (groups < 2 || total_digits < 8 || total_digits > 15) return 0;
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_paren_phone(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    if (text[pos] != '(') return 0;
    uint32_t p = pos + 1;
    int d1 = 0;
    while (p < len && is_digit((unsigned char)text[p]) && d1 < 4) { d1++; p++; }
    if (d1 < 2 || d1 > 4 || p >= len || text[p] != ')') return 0;
    p++;
    if (p < len && (text[p] == ' ' || text[p] == '-')) p++;
    int d2 = 0;
    while (p < len && is_digit((unsigned char)text[p])) { d2++; p++; }
    if (d2 < 3 || d2 > 4) return 0;
    if (p < len && (text[p] == '-' || text[p] == '.' || text[p] == ' ')) {
        p++;
        int d3 = 0;
        while (p < len && is_digit((unsigned char)text[p])) { d3++; p++; }
        if (d3 < 3 || d3 > 4) return 0;
    }
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_date_long(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    static const char *months[] = {
        "January","February","March","April","May","June",
        "July","August","September","October","November","December",
    };
    for (int m = 0; m < 12; ++m) {
        uint32_t mlen = (uint32_t)strlen(months[m]);
        if (pos + mlen > len) continue;
        if (memcmp(text + pos, months[m], mlen) != 0) continue;
        uint32_t p = pos + mlen;
        if (p >= len || text[p] != ' ') continue;
        p++;
        int day = 0, dd = 0;
        while (p < len && is_digit((unsigned char)text[p]) && dd < 2) { day = day * 10 + (text[p] - '0'); dd++; p++; }
        if (dd == 0 || day < 1 || day > 31) continue;
        if (p < len && text[p] == ',') p++;
        if (p >= len || text[p] != ' ') continue;
        p++;
        int yr = 0, yd = 0;
        while (p < len && is_digit((unsigned char)text[p]) && yd < 4) { yr = yr * 10 + (text[p] - '0'); yd++; p++; }
        if (yd != 4 || yr < 1900 || yr > 2100) continue;
        if (p < len && is_alnum((unsigned char)text[p])) continue;
        *end = p;
        return 1;
    }
    return 0;
}

static int try_match_date_cjk(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    int yr = 0, yd = 0;
    while (p < len && is_digit((unsigned char)text[p]) && yd < 4) { yr = yr * 10 + (text[p] - '0'); yd++; p++; }
    if (yd != 4 || yr < 1900 || yr > 2100) return 0;
    /* \xe5\xb9\xb4 = 年 */
    if (p + 3 > len || (unsigned char)text[p] != 0xe5 || (unsigned char)text[p+1] != 0xb9 || (unsigned char)text[p+2] != 0xb4) return 0;
    p += 3;
    int mo = 0, md = 0;
    while (p < len && is_digit((unsigned char)text[p]) && md < 2) { mo = mo * 10 + (text[p] - '0'); md++; p++; }
    if (md == 0 || mo < 1 || mo > 12) return 0;
    /* \xe6\x9c\x88 = 月 */
    if (p + 3 > len || (unsigned char)text[p] != 0xe6 || (unsigned char)text[p+1] != 0x9c || (unsigned char)text[p+2] != 0x88) return 0;
    p += 3;
    int da = 0, dd = 0;
    while (p < len && is_digit((unsigned char)text[p]) && dd < 2) { da = da * 10 + (text[p] - '0'); dd++; p++; }
    if (dd == 0 || da < 1 || da > 31) return 0;
    /* \xe6\x97\xa5 = 日 */
    if (p + 3 > len || (unsigned char)text[p] != 0xe6 || (unsigned char)text[p+1] != 0x97 || (unsigned char)text[p+2] != 0xa5) return 0;
    p += 3;
    *end = p;
    return 1;
}

static int try_match_date_euro(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    int d1 = 0, n1 = 0;
    while (p < len && is_digit((unsigned char)text[p]) && n1 < 2) { d1 = d1 * 10 + (text[p] - '0'); n1++; p++; }
    if (n1 == 0 || d1 < 1 || d1 > 31) return 0;
    char sep = 0;
    if (p < len && (text[p] == '.' || text[p] == '/')) { sep = text[p]; p++; } else return 0;
    int d2 = 0, n2 = 0;
    while (p < len && is_digit((unsigned char)text[p]) && n2 < 2) { d2 = d2 * 10 + (text[p] - '0'); n2++; p++; }
    if (n2 == 0 || d2 < 1 || d2 > 12) return 0;
    if (p >= len || text[p] != sep) return 0;
    p++;
    int yr = 0, n3 = 0;
    while (p < len && is_digit((unsigned char)text[p]) && n3 < 4) { yr = yr * 10 + (text[p] - '0'); n3++; p++; }
    if (n3 != 4 || yr < 1900 || yr > 2100) return 0;
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_ssn(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    int d1 = 0;
    while (p < len && is_digit((unsigned char)text[p])) { d1++; p++; }
    if (d1 != 3 || p >= len || text[p] != '-') return 0;
    p++;
    int d2 = 0;
    while (p < len && is_digit((unsigned char)text[p])) { d2++; p++; }
    if (d2 != 2 || p >= len || text[p] != '-') return 0;
    p++;
    int d3 = 0;
    while (p < len && is_digit((unsigned char)text[p])) { d3++; p++; }
    if (d3 != 4) return 0;
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_chinese_id(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    int d = 0;
    while (p < len && is_digit((unsigned char)text[p])) { d++; p++; }
    if (d == 18) {
        if (p < len && is_alnum((unsigned char)text[p])) return 0;
        *end = p;
        return 1;
    }
    if (d == 17 && p < len && (text[p] == 'X' || text[p] == 'x')) {
        p++;
        if (p < len && is_alnum((unsigned char)text[p])) return 0;
        *end = p;
        return 1;
    }
    return 0;
}

static int try_match_credit_card(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    int digits = 0, groups = 0;
    while (p < len && digits < 20) {
        if (is_digit((unsigned char)text[p])) {
            int gd = 0;
            while (p < len && is_digit((unsigned char)text[p])) { gd++; p++; digits++; }
            groups++;
            if (p < len && (text[p] == ' ' || text[p] == '-') && is_digit((unsigned char)text[p < len - 1 ? p + 1 : p])) p++;
            else break;
        } else break;
    }
    if (digits < 13 || digits > 19 || groups < 2) return 0;
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_iban(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    if (p + 4 >= len) return 0;
    if (!is_upper((unsigned char)text[p]) || !is_upper((unsigned char)text[p+1])) return 0;
    if (!is_digit((unsigned char)text[p+2]) || !is_digit((unsigned char)text[p+3])) return 0;
    p += 4;
    int alnum_count = 4, groups = 1;
    while (p < len && alnum_count < 34) {
        if (text[p] == ' ') {
            p++;
            if (p >= len || !is_alnum((unsigned char)text[p])) break;
            groups++;
        } else if (is_alnum((unsigned char)text[p])) {
            alnum_count++; p++;
        } else break;
    }
    if (alnum_count < 15 || groups < 3) return 0;
    if (p < len && is_alnum((unsigned char)text[p])) return 0;
    *end = p;
    return 1;
}

static int try_match_api_key(const char *text, uint32_t pos, uint32_t len, uint32_t *end) {
    struct { const char *prefix; int plen; int min_total; } keys[] = {
        {"sk-",   3, 20}, {"pk-",   3, 20}, {"sk_live_", 8, 20}, {"sk_test_", 8, 20},
        {"pk_live_", 8, 20}, {"pk_test_", 8, 20},
        {"ghp_",  4, 20}, {"gho_",  4, 20}, {"ghs_",  4, 20}, {"ghr_",  4, 20},
        {"github_pat_", 11, 30},
        {"AKIA",  4, 20}, {"ABIA",  4, 20}, {"ACCA",  4, 20}, {"ASIA",  4, 20},
        {"xoxb-", 5, 20}, {"xoxp-", 5, 20}, {"xoxs-", 5, 20}, {"xoxa-", 5, 20},
        {"eyJ",   3, 30},
        {"glpat-", 6, 20}, {"npd_",  4, 20},
        {"Bearer ", 7, 20},
    };
    int nk = (int)(sizeof(keys) / sizeof(keys[0]));
    for (int k = 0; k < nk; ++k) {
        uint32_t plen = (uint32_t)keys[k].plen;
        if (pos + plen > len) continue;
        if (memcmp(text + pos, keys[k].prefix, plen) != 0) continue;
        uint32_t p = pos + plen;
        while (p < len && (is_alnum((unsigned char)text[p]) || text[p] == '-' || text[p] == '_' || text[p] == '.' || text[p] == ':' || text[p] == '/'))
            p++;
        if ((int)(p - pos) >= keys[k].min_total) {
            *end = p;
            return 1;
        }
    }
    return 0;
}

static int match_scheme(const char *text, uint32_t pos, uint32_t len, uint32_t *scheme_end, const char **label_out) {
    struct { const char *prefix; int plen; const char *label; } schemes[] = {
        {"https://",   8, "private_url"},
        {"http://",    7, "private_url"},
        {"ftps://",    7, "private_url"},
        {"ftp://",     6, "private_url"},
        {"sftp://",    7, "private_url"},
        {"ssh://",     6, "private_url"},
        {"webdavs://",10, "private_url"},
        {"webdav://",  9, "private_url"},
        {"davs://",    7, "private_url"},
        {"dav://",     6, "private_url"},
        {"smb://",     6, "private_url"},
        {"nfs://",     6, "private_url"},
        {"ldaps://",   8, "private_url"},
        {"ldap://",    7, "private_url"},
        {"vnc://",     6, "private_url"},
        {"rdp://",     6, "private_url"},
        {"mailto:",    7, "private_email"},
        {"tel:",       4, "private_phone"},
        {"s3://",      5, "secret"},
        {"gs://",      5, "secret"},
    };
    int n = (int)(sizeof(schemes) / sizeof(schemes[0]));
    for (int i = 0; i < n; ++i) {
        uint32_t plen = (uint32_t)schemes[i].plen;
        if (pos + plen <= len && memcmp(text + pos, schemes[i].prefix, plen) == 0) {
            *scheme_end = pos + plen;
            *label_out = schemes[i].label;
            return 1;
        }
    }
    return 0;
}

static int try_match_ip(const char *s, uint32_t pos, uint32_t len, uint32_t *end) {
    uint32_t p = pos;
    for (int oct = 0; oct < 4; ++oct) {
        if (p >= len || !is_digit((unsigned char)s[p])) return 0;
        int v = 0, digits = 0;
        while (p < len && is_digit((unsigned char)s[p]) && digits < 3) { v = v * 10 + (s[p] - '0'); p++; digits++; }
        if (v > 255) return 0;
        if (oct < 3) { if (p >= len || s[p] != '.') return 0; p++; }
    }
    if (p < len && (is_alnum((unsigned char)s[p]) || s[p] == '.')) return 0;
    *end = p;
    return 1;
}

static void ensure_span_capacity(PFSpan **spans, int *cap, int count) {
    if (count < *cap) return;
    *cap *= 2;
    *spans = (PFSpan *)xreallocarray(*spans, (size_t)*cap, sizeof(PFSpan));
}

static int span_overlaps(PFSpan *spans, int count, uint32_t start, uint32_t end) {
    for (int s = 0; s < count; ++s) {
        if (spans[s].start < end && spans[s].end > start) return 1;
    }
    return 0;
}

static void add_span_if_new(const char *text, PFSpan **spans, int *count, int *cap,
                            uint32_t start, uint32_t end, const char *label) {
    (void)text;
    if (span_overlaps(*spans, *count, start, end)) return;
    ensure_span_capacity(spans, cap, *count);
    (*spans)[*count].start = start;
    (*spans)[*count].end = end;
    copy_label((*spans)[*count].label, label);
    (*count)++;
}

static void regex_supplement(const char *text, PFSpan **spans, int *count, int *cap) {
    uint32_t len = (uint32_t)strlen(text);
    for (uint32_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];

        /* ---- International phone: +CC ... ---- */
        if (c == '+' && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            uint32_t pe;
            if (try_match_intl_phone(text, i, len, &pe)) {
                add_span_if_new(text, spans, count, cap, i, pe, "private_phone");
                i = pe - 1;
                continue;
            }
        }

        /* ---- Parenthesized phone: (NNN) NNN-NNNN ---- */
        if (c == '(' && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            uint32_t pe;
            if (try_match_paren_phone(text, i, len, &pe)) {
                add_span_if_new(text, spans, count, cap, i, pe, "private_phone");
                i = pe - 1;
                continue;
            }
        }

        /* ---- API keys / secrets ---- */
        if ((c == 's' || c == 'p' || c == 'g' || c == 'A' || c == 'x' || c == 'e' || c == 'n' || c == 'B')
            && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            uint32_t ke;
            if (try_match_api_key(text, i, len, &ke)) {
                add_span_if_new(text, spans, count, cap, i, ke, "secret");
                i = ke - 1;
                continue;
            }
        }

        /* ---- Long-form English date: January 15, 2024 ---- */
        if (is_upper(c) && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            uint32_t de;
            if (try_match_date_long(text, i, len, &de)) {
                add_span_if_new(text, spans, count, cap, i, de, "private_date");
                i = de - 1;
                continue;
            }
        }

        /* ---- CJK date: 2024年5月27日 ---- */
        if (is_digit(c) && (i == 0 || !is_digit((unsigned char)text[i - 1]))) {
            uint32_t de;
            if (try_match_date_cjk(text, i, len, &de)) {
                add_span_if_new(text, spans, count, cap, i, de, "private_date");
                i = de - 1;
                continue;
            }
        }

        if (c == '@' && i > 0) {
            uint32_t es = i - 1;
            while (es > 0 && (is_alnum((unsigned char)text[es - 1]) || text[es - 1] == '.' || text[es - 1] == '_' || text[es - 1] == '%' || text[es - 1] == '+' || text[es - 1] == '-'))
                es--;
            if (es == i) continue;
            uint32_t ee = i + 1;
            while (ee < len && (is_alnum((unsigned char)text[ee]) || text[ee] == '.' || text[ee] == '-'))
                ee++;
            while (ee > i + 1 && (text[ee - 1] == '.' || text[ee - 1] == '-')) ee--;
            if (ee == i + 1) continue;
            uint32_t dot = 0;
            for (uint32_t j = i + 1; j < ee; ++j) if (text[j] == '.') dot = j;
            if (!dot || ee - dot < 3) continue;
            int already = 0;
            for (int s = 0; s < *count; ++s) {
                if ((*spans)[s].start <= es && (*spans)[s].end >= ee) { already = 1; break; }
            }
            if (already) continue;
            ensure_span_capacity(spans, cap, *count);
            (*spans)[*count].start = es; (*spans)[*count].end = ee;
            copy_label((*spans)[*count].label, "private_email");
            (*count)++;
        }
        if (is_digit(c) && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            uint32_t ip_end;
            if (try_match_ip(text, i, len, &ip_end)) {
                add_span_if_new(text, spans, count, cap, i, ip_end, "secret");
                i = ip_end - 1;
                continue;
            }
            /* European date: DD.MM.YYYY or DD/MM/YYYY */
            {
                uint32_t de;
                if (try_match_date_euro(text, i, len, &de)) {
                    add_span_if_new(text, spans, count, cap, i, de, "private_date");
                    i = de - 1;
                    continue;
                }
            }
            /* SSN: NNN-NN-NNNN */
            {
                uint32_t se;
                if (try_match_ssn(text, i, len, &se)) {
                    add_span_if_new(text, spans, count, cap, i, se, "account_number");
                    i = se - 1;
                    continue;
                }
            }
            /* Chinese national ID: 18 digits (last may be X) */
            {
                uint32_t ce;
                if (try_match_chinese_id(text, i, len, &ce)) {
                    add_span_if_new(text, spans, count, cap, i, ce, "account_number");
                    i = ce - 1;
                    continue;
                }
            }
            /* Credit card: 13-19 digits in groups */
            {
                uint32_t ce;
                if (try_match_credit_card(text, i, len, &ce)) {
                    add_span_if_new(text, spans, count, cap, i, ce, "account_number");
                    i = ce - 1;
                    continue;
                }
            }
            int d1 = 0;
            while (i + d1 < len && is_digit((unsigned char)text[i + d1])) d1++;
            /* EIN: NN-NNNNNNN */
            if (d1 == 2 && i + d1 < len && text[i + d1] == '-') {
                uint32_t dp = i + (uint32_t)d1 + 1;
                int d2 = 0;
                while (dp + d2 < len && is_digit((unsigned char)text[dp + d2])) d2++;
                if (d2 == 7 && (dp + d2 >= len || !is_alnum((unsigned char)text[dp + d2]))) {
                    uint32_t ein_end = dp + (uint32_t)d2;
                    add_span_if_new(text, spans, count, cap, i, ein_end, "account_number");
                    i = ein_end - 1;
                    continue;
                }
            }
            /* US street address: N+ Street, ST NNNNN */
            {
                int addr_found = 0;
                if (d1 >= 1 && d1 <= 5 && i + (uint32_t)d1 < len && text[i + (uint32_t)d1] == ' ') {
                    for (uint32_t j = i + (uint32_t)d1 + 1; j + 8 < len && j < i + 250; ++j) {
                        if (text[j] == ',' && text[j+1] == ' ' &&
                            is_upper((unsigned char)text[j+2]) && is_upper((unsigned char)text[j+3]) &&
                            !is_upper((unsigned char)text[j+4]) &&
                            text[j+4] == ' ' && is_digit((unsigned char)text[j+5])) {
                            uint32_t zs = j + 5;
                            int zd = 0;
                            while (zs + (uint32_t)zd < len && is_digit((unsigned char)text[zs + (uint32_t)zd])) zd++;
                            if (zd == 5 && (zs + 5 >= len || !is_alnum((unsigned char)text[zs + 5]))) {
                                uint32_t addr_end = zs + 5;
                                add_span_if_new(text, spans, count, cap, i, addr_end, "private_address");
                                i = addr_end - 1;
                                addr_found = 1;
                                break;
                            }
                        }
                    }
                }
                if (addr_found) continue;
            }
        }
        {
            uint32_t scheme_end;
            const char *url_label;
            if (match_scheme(text, i, len, &scheme_end, &url_label)) {
                uint32_t se = scheme_end;
                while (se < len && is_url_body_char((unsigned char)text[se])) se++;
                while (se > scheme_end && (text[se - 1] == '.' || text[se - 1] == ',' || text[se - 1] == ')' || text[se - 1] == ';'))
                    se--;
                if (se > scheme_end) {
                    add_span_if_new(text, spans, count, cap, i, se, url_label);
                    i = se - 1;
                    continue;
                }
            }
        }
        if (is_upper(c) && (i == 0 || !is_alnum((unsigned char)text[i - 1]))) {
            /* IBAN: 2 upper + 2 digit + groups of alnum */
            {
                uint32_t ie;
                if (try_match_iban(text, i, len, &ie)) {
                    add_span_if_new(text, spans, count, cap, i, ie, "account_number");
                    i = ie - 1;
                    continue;
                }
            }
            uint32_t se = i;
            int uppers = 0, digits = 0, dashes = 0;
            while (se < len && (is_upper((unsigned char)text[se]) || is_digit((unsigned char)text[se]) || text[se] == '-')) {
                if (is_upper((unsigned char)text[se])) uppers++;
                else if (is_digit((unsigned char)text[se])) digits++;
                else dashes++;
                se++;
            }
            uint32_t slen = se - i;
            int matched = 0;
            if (slen >= 8 && uppers >= 2 && dashes >= 1 && digits >= 2 && (se >= len || !is_alnum((unsigned char)text[se]))) {
                matched = 1;
            }
            if (!matched && slen >= 8 && slen <= 11 && uppers >= 6 && digits <= 5 && dashes == 0 && (se >= len || !is_alnum((unsigned char)text[se]))) {
                matched = 1;
            }
            if (matched) {
                add_span_if_new(text, spans, count, cap, i, se, "account_number");
                i = se - 1;
                continue;
            }
        }
        if (is_alnum_dot_dash(c) && i + 4 < len) {
            uint32_t hs = i;
            while (hs > 0 && is_alnum_dot_dash((unsigned char)text[hs - 1])) hs--;
            if (hs != i) { i++; continue; }
            uint32_t he = i;
            while (he < len && is_alnum_dot_dash((unsigned char)text[he])) he++;
            uint32_t hlen = he - hs;
            if (hlen > 8 && hlen < 100) {
                int dots = 0;
                for (uint32_t j = hs; j < he; ++j) { if (text[j] == '.') dots++; }
                const char *p = text + hs;
                uint32_t plen = hlen;
                static const struct { const char *tld; int tlen; } tlds[] = {
                    {".com",4},{".net",4},{".org",4},{".edu",4},{".gov",4},{".mil",4},
                    {".int",4},{".info",5},{".biz",4},{".name",5},{".pro",4},{".mobi",5},
                    {".io",3},{".co",3},{".ai",3},{".de",3},{".jp",3},{".fr",3},
                    {".cn",3},{".kr",3},{".ru",3},{".br",3},{".in",3},{".uk",3},
                    {".au",3},{".ca",3},{".es",3},{".it",3},{".nl",3},{".se",3},
                    {".no",3},{".dk",3},{".fi",3},{".pl",3},{".cz",3},{".tw",3},
                    {".hk",3},{".sg",3},{".my",3},{".th",3},{".vn",3},{".id",3},
                    {".ph",3},{".za",3},{".mx",3},{".ar",3},{".cl",3},{".pt",3},
                    {".at",3},{".ch",3},{".be",3},{".ie",3},{".nz",3},{".il",3},
                };
                int ntlds = (int)(sizeof(tlds) / sizeof(tlds[0]));
                int tld_match = 0;
                if (plen > 9 && dots >= 2) {
                    for (int t = 0; t < ntlds; ++t) {
                        if ((int)plen >= tlds[t].tlen && strncmp(p + plen - tlds[t].tlen, tlds[t].tld, (size_t)tlds[t].tlen) == 0) {
                            tld_match = 1; break;
                        }
                    }
                }
                if (tld_match) {
                    add_span_if_new(text, spans, count, cap, hs, he, "private_url");
                }
            }
            i = he - 1;
        }
    }
}

static void tighten_email_spans(const char *text, PFSpan *spans, int count) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(spans[i].label, "private_email") != 0) continue;
        uint32_t at_pos = 0;
        for (uint32_t j = spans[i].start; j < spans[i].end; ++j) {
            if (text[j] == '@') { at_pos = j; break; }
        }
        if (!at_pos) continue;
        uint32_t ns = at_pos;
        while (ns > spans[i].start && (is_alnum((unsigned char)text[ns - 1]) || text[ns-1] == '.' ||
               text[ns-1] == '_' || text[ns-1] == '%' || text[ns-1] == '+' || text[ns-1] == '-'))
            ns--;
        uint32_t ne = at_pos + 1;
        while (ne < spans[i].end && (is_alnum((unsigned char)text[ne]) || text[ne] == '.' || text[ne] == '-'))
            ne++;
        while (ne > at_pos + 1 && (text[ne - 1] == '.' || text[ne - 1] == '-')) ne--;
        if (ns < at_pos && ne > at_pos + 1) {
            spans[i].start = ns;
            spans[i].end = ne;
        }
    }
}

static void sort_spans(PFSpan *spans, int count) {
    for (int i = 1; i < count; ++i) {
        PFSpan tmp = spans[i];
        int j = i - 1;
        while (j >= 0 && (spans[j].start > tmp.start || (spans[j].start == tmp.start && spans[j].end < tmp.end))) {
            spans[j + 1] = spans[j]; j--;
        }
        spans[j + 1] = tmp;
    }
}

static void trim_span_whitespace(const char *text, PFSpan *span) {
    while (span->start < span->end && byte_is_space((unsigned char)text[span->start])) span->start++;
    while (span->end > span->start && byte_is_space((unsigned char)text[span->end - 1])) span->end--;
}

static int normalize_spans(PFSpan *spans, int count, const char *text) {
    if (text) {
        for (int i = 0; i < count; ++i) trim_span_whitespace(text, &spans[i]);
    }
    sort_spans(spans, count);
    int out = 0;
    uint32_t last_end = 0;
    for (int i = 0; i < count; ++i) {
        if (spans[i].start >= spans[i].end) continue;
        if (spans[i].start < last_end) continue;
        if (out != i) spans[out] = spans[i];
        last_end = spans[out].end;
        out++;
    }
    return out;
}

static int pf_predict_segment(PFModel *model, const char *text, PFSpan **spans_out, int *count_out) {
    PFToken *tokens = NULL;
    int ntok = 0;
    if (pf_tokenize(model, text, &tokens, &ntok) != 0) return -1;
    if (ntok == 0) {
        free(tokens);
        *spans_out = NULL;
        *count_out = 0;
        return 0;
    }
    if (ensure_buffers(model, ntok) != 0) {
        free(tokens); return -1;
    }
    float *logits = (float *)xmalloc((size_t)ntok * PF_LABELS * sizeof(float));
    if (forward_tokens(model, tokens, ntok, logits) != 0) {
        free(logits); free(tokens); return -1;
    }
    int *labels = viterbi_decode(model, logits, ntok);
    free(logits);
    int rc = labels_to_spans(tokens, labels, ntok, spans_out, count_out);
    if (rc == 0) {
        int out = 0;
        for (int i = 0; i < *count_out; ++i) {
            trim_span_whitespace(text, &(*spans_out)[i]);
            if ((*spans_out)[i].end > (*spans_out)[i].start) {
                if (out != i) (*spans_out)[out] = (*spans_out)[i];
                ++out;
            }
        }
        *count_out = out;
    }
    free(tokens); free(labels);
    return rc;
}

static void append_spans(PFSpan **all, int *total, int *cap, PFSpan *segs, int n, uint32_t offset) {
    for (int i = 0; i < n; ++i) {
        segs[i].start += offset;
        segs[i].end += offset;
        if (*total >= *cap) {
            *cap *= 2;
            *all = (PFSpan *)xreallocarray(*all, (size_t)*cap, sizeof(PFSpan));
        }
        (*all)[(*total)++] = segs[i];
    }
    pf_free(segs);
}

int pf_predict(PFModel *model, const char *text, PFSpan **spans_out, int *count_out) {
    if (!model || !text || !spans_out || !count_out) return -1;
    size_t text_len = strlen(text);
    if (text_len < PF_PARAGRAPH_THRESHOLD) {
        int rc = pf_predict_segment(model, text, spans_out, count_out);
        if (rc == 0) {
            int cap = *count_out + 16;
            *spans_out = (PFSpan *)xreallocarray(*spans_out, (size_t)cap, sizeof(PFSpan));
            regex_supplement(text, spans_out, count_out, &cap);
            tighten_email_spans(text, *spans_out, *count_out);
            *count_out = normalize_spans(*spans_out, *count_out, text);
        }
        return rc;
    }
    int total_spans = 0, spans_cap = 64;
    PFSpan *all_spans = (PFSpan *)xmalloc((size_t)spans_cap * sizeof(PFSpan));
    size_t pos = 0;
    while (pos < text_len) {
        while (pos < text_len && byte_is_space((unsigned char)text[pos])) pos++;
        if (pos >= text_len) break;
        size_t seg_start = pos;
        size_t seg_end = seg_start;
        while (pos < text_len) {
            if (text[pos] == '\n' || text[pos] == '\r') {
                seg_end = pos; pos++; break;
            }
            if ((text[pos] == '?' || text[pos] == '!') && pos + 1 < text_len && text[pos + 1] == ' ') {
                seg_end = pos + 1; pos += 2; break;
            }
            if (text[pos] == '.' && pos + 2 < text_len && text[pos + 1] == ' ' && is_upper((unsigned char)text[pos + 2])) {
                int abbr = 0;
                size_t wb = pos;
                while (wb > seg_start && ((text[wb-1]>='A' && text[wb-1]<='Z') || (text[wb-1]>='a' && text[wb-1]<='z'))) wb--;
                size_t wl = pos - wb;
                if (wl >= 1 && wl <= 4) {
                    const char *w = text + wb;
                    if (wl == 1 ||
                        (wl == 2 && (memcmp(w,"Dr",2)==0||memcmp(w,"Mr",2)==0||memcmp(w,"Ms",2)==0||
                                     memcmp(w,"St",2)==0||memcmp(w,"Jr",2)==0||memcmp(w,"Sr",2)==0||
                                     memcmp(w,"vs",2)==0||memcmp(w,"No",2)==0)) ||
                        (wl == 3 && (memcmp(w,"Mrs",3)==0||memcmp(w,"Ave",3)==0||memcmp(w,"Inc",3)==0)) ||
                        (wl == 4 && (memcmp(w,"Prof",4)==0||memcmp(w,"Blvd",4)==0||memcmp(w,"Corp",4)==0)))
                        abbr = 1;
                }
                if (!abbr) { seg_end = pos + 1; pos += 2; break; }
            }
            pos++;
        }
        if (pos >= text_len && seg_end <= seg_start) seg_end = text_len;
        while (seg_end > seg_start && byte_is_space((unsigned char)text[seg_end - 1])) seg_end--;
        size_t seg_len = seg_end - seg_start;
        if (seg_len == 0) continue;
        char *seg = (char *)xmalloc(seg_len + 2);
        memcpy(seg, text + seg_start, seg_len);
        seg[seg_len] = '\n'; seg[seg_len + 1] = '\0';
        PFSpan *seg_spans = NULL;
        int seg_count = 0;
        int rc = pf_predict_segment(model, seg, &seg_spans, &seg_count);
        free(seg);
        if (rc != 0) { free(all_spans); return rc; }
        append_spans(&all_spans, &total_spans, &spans_cap, seg_spans, seg_count, (uint32_t)seg_start);
    }
    regex_supplement(text, &all_spans, &total_spans, &spans_cap);
    tighten_email_spans(text, all_spans, total_spans);
    total_spans = normalize_spans(all_spans, total_spans, NULL);
    *spans_out = all_spans;
    *count_out = total_spans;
    return 0;
}

int pf_predict_raw(PFModel *model, const char *text, PFSpan **spans_out, int *count_out) {
    if (!model || !text || !spans_out || !count_out) return -1;
    size_t text_len = strlen(text);
    if (text_len < PF_PARAGRAPH_THRESHOLD) {
        int rc = pf_predict_segment(model, text, spans_out, count_out);
        if (rc == 0) *count_out = normalize_spans(*spans_out, *count_out, text);
        return rc;
    }
    int total_spans = 0, spans_cap = 64;
    PFSpan *all_spans = (PFSpan *)xmalloc((size_t)spans_cap * sizeof(PFSpan));
    size_t pos = 0;
    while (pos < text_len) {
        while (pos < text_len && byte_is_space((unsigned char)text[pos])) pos++;
        if (pos >= text_len) break;
        size_t seg_start = pos;
        size_t seg_end = seg_start;
        while (pos < text_len) {
            if (text[pos] == '\n' || text[pos] == '\r') { seg_end = pos; pos++; break; }
            if ((text[pos] == '?' || text[pos] == '!') && pos + 1 < text_len && text[pos + 1] == ' ') { seg_end = pos + 1; pos += 2; break; }
            if (text[pos] == '.' && pos + 2 < text_len && text[pos + 1] == ' ' && is_upper((unsigned char)text[pos + 2])) {
                int abbr = 0;
                size_t wb = pos;
                while (wb > seg_start && ((text[wb-1]>='A' && text[wb-1]<='Z') || (text[wb-1]>='a' && text[wb-1]<='z'))) wb--;
                size_t wl = pos - wb;
                if (wl >= 1 && wl <= 4) {
                    const char *w = text + wb;
                    if (wl == 1 ||
                        (wl == 2 && (memcmp(w,"Dr",2)==0||memcmp(w,"Mr",2)==0||memcmp(w,"Ms",2)==0||
                                     memcmp(w,"St",2)==0||memcmp(w,"Jr",2)==0||memcmp(w,"Sr",2)==0||
                                     memcmp(w,"vs",2)==0||memcmp(w,"No",2)==0)) ||
                        (wl == 3 && (memcmp(w,"Mrs",3)==0||memcmp(w,"Ave",3)==0||memcmp(w,"Inc",3)==0)) ||
                        (wl == 4 && (memcmp(w,"Prof",4)==0||memcmp(w,"Blvd",4)==0||memcmp(w,"Corp",4)==0)))
                        abbr = 1;
                }
                if (!abbr) { seg_end = pos + 1; pos += 2; break; }
            }
            pos++;
        }
        if (pos >= text_len && seg_end <= seg_start) seg_end = text_len;
        while (seg_end > seg_start && byte_is_space((unsigned char)text[seg_end - 1])) seg_end--;
        size_t seg_len = seg_end - seg_start;
        if (seg_len == 0) continue;
        char *seg = (char *)xmalloc(seg_len + 2);
        memcpy(seg, text + seg_start, seg_len);
        seg[seg_len] = '\n'; seg[seg_len + 1] = '\0';
        PFSpan *seg_spans = NULL;
        int seg_count = 0;
        int rc = pf_predict_segment(model, seg, &seg_spans, &seg_count);
        free(seg);
        if (rc != 0) { free(all_spans); return rc; }
        append_spans(&all_spans, &total_spans, &spans_cap, seg_spans, seg_count, (uint32_t)seg_start);
    }
    total_spans = normalize_spans(all_spans, total_spans, NULL);
    *spans_out = all_spans;
    *count_out = total_spans;
    return 0;
}

int pf_redact(PFModel *model, const char *text, char **redacted_out) {
    if (!model || !text || !redacted_out) return -1;
    PFSpan *spans = NULL;
    int count = 0;
    if (pf_predict(model, text, &spans, &count) != 0) return -1;
    size_t text_len = strlen(text);
    size_t out_len = text_len + 1;
    const char *placeholder = "<REDACTED>";
    size_t ph_len = strlen(placeholder);
    for (int i = 0; i < count; ++i) {
        if (spans[i].end > spans[i].start) {
            size_t span_len = (size_t)(spans[i].end - spans[i].start);
            if (ph_len > span_len) {
                if (out_len > SIZE_MAX - (ph_len - span_len)) { pf_free(spans); return -1; }
                out_len += ph_len - span_len;
            } else {
                out_len -= span_len - ph_len;
            }
        }
    }
    char *out = (char *)xmalloc(out_len);
    size_t w = 0;
    uint32_t cursor = 0;
    for (int i = 0; i < count; ++i) {
        if (spans[i].start < cursor) continue;
        size_t n = spans[i].start - cursor;
        memcpy(out + w, text + cursor, n);
        w += n;
        memcpy(out + w, placeholder, ph_len);
        w += ph_len;
        cursor = spans[i].end;
    }
    size_t tail = text_len - cursor;
    memcpy(out + w, text + cursor, tail);
    w += tail;
    out[w] = 0;
    pf_free(spans);
    *redacted_out = out;
    return 0;
}

PFModel *pf_load_model(const char *model_dir) {
    PFModel *m = (PFModel *)xcalloc(1, sizeof(PFModel));
    m->model_dir = xstrdup(model_dir);
    char *vocab_path = join_path(model_dir, "tokenizer.json");
    char *calibration_path = join_path(model_dir, "viterbi_calibration.json");
    m->weights_path = join_path(model_dir, "model.safetensors");
    if (load_vocab(m, vocab_path) != 0 || load_index(m, m->weights_path) != 0 || load_viterbi_calibration(m, calibration_path) != 0) {
        free(vocab_path); free(calibration_path); pf_free_model(m); return NULL;
    }
    qsort(m->tensors, (size_t)m->tensor_count, sizeof(Tensor), tensor_compare);
#ifdef _MSC_VER
    fopen_s(&m->weights_file, m->weights_path, "rb");
#else
    m->weights_file = fopen(m->weights_path, "rb");
#endif
    if (!m->weights_file) {
        fprintf(stderr, "failed to open weights file %s\n", m->weights_path);
        free(vocab_path); free(calibration_path); pf_free_model(m); return NULL;
    }
    if (map_weights(m) != 0) {
        fprintf(stderr, "warning: memory mapping failed for %s; falling back to buffered reads\n", m->weights_path);
    }
    init_viterbi(m);
    free(vocab_path);
    free(calibration_path);
    return m;
}

void pf_free(void *ptr) {
    free(ptr);
}

void pf_free_model(PFModel *m) {
    if (!m) return;
    if (m->weights_file) fclose(m->weights_file);
    if (m->tensors) {
        for (int i = 0; i < m->tensor_count; ++i) {
            free(m->tensors[i].name);
            if (m->tensors[i].data && (!m->weights_map || (uint8_t *)m->tensors[i].data < m->weights_map || (uint8_t *)m->tensors[i].data >= m->weights_map + m->weights_map_size)) {
                free(m->tensors[i].data);
            }
        }
        free(m->tensors);
    }
    unmap_weights(m);
    for (int i = 0; i < PF_VOCAB_SIZE; ++i) free(m->id_to_vocab[i]);
    free(m->specials);
    trie_free(m->trie);
    free(m->model_dir);
    free(m->weights_path);
    free(m->x);
    free(m->norm);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->moe);
    free(m->router_logits);
    free(m->moe_expert);
    free(m->moe_token);
    free(m->moe_sorted_token);
    free(m->moe_weight);
    free(m->moe_sorted_weight);
    free(m->moe_thread_scratch);
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->logits);
    free(m->transition);
    free(m->start_scores);
    free(m->end_scores);
    free(m);
}
