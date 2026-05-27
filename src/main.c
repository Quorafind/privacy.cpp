#include "privacy_filter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *pf_memmem(const void *hay, size_t hlen, const void *needle, size_t nlen) {
    if (nlen == 0) return (void *)hay;
    if (nlen > hlen) return NULL;
    const char *h = (const char *)hay;
    const char *n = (const char *)needle;
    for (size_t i = 0; i <= hlen - nlen; i++) {
        if (memcmp(h + i, n, nlen) == 0) return (void *)(h + i);
    }
    return NULL;
}

static char *read_all_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "failed to open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    return buf;
}

static char *read_all_stdin(void) {
    size_t cap = 4096;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;
    int c;
    while ((c = getchar()) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *next = (char *)realloc(buf, cap);
            if (!next) { free(buf); return NULL; }
            buf = next;
        }
        buf[len++] = (char)c;
    }
    buf[len] = 0;
    return buf;
}

static char *copy_string(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
}

static char *join_path_cli(const char *a, const char *b) {
    size_t alen = strlen(a);
    size_t blen = strlen(b);
    int need_sep = alen > 0 && a[alen - 1] != '/' && a[alen - 1] != '\\';
#ifdef _WIN32
    char sep = '\\';
#else
    char sep = '/';
#endif
    char *out = (char *)malloc(alen + (size_t)need_sep + blen + 1);
    if (!out) return NULL;
    memcpy(out, a, alen);
    if (need_sep) out[alen++] = sep;
    memcpy(out + alen, b, blen + 1);
    return out;
}

static char *path_dirname_cli(const char *path) {
    const char *last = NULL;
    for (const char *p = path; p && *p; ++p) {
        if (*p == '/' || *p == '\\') last = p;
    }
    if (!last) return copy_string(".");

    size_t len = (size_t)(last - path);
#ifdef _WIN32
    if (len == 2 && path[1] == ':') len = 3;
#else
    if (len == 0) len = 1;
#endif
    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;
    memcpy(out, path, len);
    out[len] = 0;
    return out;
}

static int file_exists_cli(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static int model_dir_ready(const char *dir) {
    char *tokenizer = join_path_cli(dir, "tokenizer.json");
    char *weights = join_path_cli(dir, "model.safetensors");
    int ok = tokenizer && weights && file_exists_cli(tokenizer) && file_exists_cli(weights);
    free(tokenizer);
    free(weights);
    return ok;
}

static char *try_model_dir(const char *dir) {
    return model_dir_ready(dir) ? copy_string(dir) : NULL;
}

static char *try_model_under(const char *base, const char *relative) {
    char *dir = join_path_cli(base, relative);
    if (!dir) return NULL;
    if (model_dir_ready(dir)) return dir;
    free(dir);
    return NULL;
}

static char *getenv_copy_cli(const char *name) {
#ifdef _MSC_VER
    char *value = NULL;
    size_t len = 0;
    if (_dupenv_s(&value, &len, name) != 0 || !value || len == 0 || value[0] == 0) {
        free(value);
        return NULL;
    }
    return value;
#else
    const char *value = getenv(name);
    return value && value[0] ? copy_string(value) : NULL;
#endif
}

static char *resolve_model_dir(const char *requested, const char *argv0) {
    if (requested && requested[0]) return copy_string(requested);

    char *env = getenv_copy_cli("PRIVACY_CPP_MODEL");
    if (env) return env;
    env = getenv_copy_cli("PRIVACY_FILTER_MODEL");
    if (env) return env;

    char *dir = try_model_dir("models/privacy-ner-v1");
    if (dir) return dir;
    dir = try_model_dir("model");
    if (dir) return dir;

    char *exe_dir = path_dirname_cli(argv0 && argv0[0] ? argv0 : ".");
    if (exe_dir) {
        dir = try_model_under(exe_dir, "models/privacy-ner-v1");
        if (!dir) dir = try_model_under(exe_dir, "model");
        free(exe_dir);
        if (dir) return dir;
    }

#ifdef _WIN32
    char *user_model_root = getenv_copy_cli("LOCALAPPDATA");
#else
    char *user_model_root = getenv_copy_cli("HOME");
#endif
    if (user_model_root) {
#ifdef _WIN32
        char *app_dir = join_path_cli(user_model_root, "privacy.cpp");
#else
        char *cache_dir = join_path_cli(user_model_root, ".cache");
        char *app_dir = cache_dir ? join_path_cli(cache_dir, "privacy.cpp") : NULL;
        free(cache_dir);
#endif
        free(user_model_root);
        if (app_dir) {
            dir = try_model_under(app_dir, "models/privacy-ner-v1");
            free(app_dir);
            if (dir) return dir;
        }
    }

    return NULL;
}

static void print_model_help(FILE *out) {
    fputs("Model directory must contain at least:\n", out);
    fputs("  tokenizer.json\n", out);
    fputs("  model.safetensors\n", out);
    fputs("Optional files:\n", out);
    fputs("  viterbi_calibration.json\n", out);
    fputs("\nDefault lookup order:\n", out);
    fputs("  --model DIR\n", out);
    fputs("  PRIVACY_CPP_MODEL or PRIVACY_FILTER_MODEL\n", out);
    fputs("  ./models/privacy-ner-v1\n", out);
    fputs("  ./model\n", out);
    fputs("  <executable-dir>/models/privacy-ner-v1\n", out);
    fputs("  <executable-dir>/model\n", out);
#ifdef _WIN32
    fputs("  %LOCALAPPDATA%\\privacy.cpp\\models\\privacy-ner-v1\n", out);
#else
    fputs("  ~/.cache/privacy.cpp/models/privacy-ner-v1\n", out);
#endif
}

static void fprint_json_str(FILE *f, const char *s, unsigned start, unsigned end) {
    fputc('"', f);
    for (unsigned i = start; i < end && s[i]; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c == '\n') fputs("\\n", f);
        else if (c == '\r') fputs("\\r", f);
        else if (c == '\t') fputs("\\t", f);
        else if (c < 0x20) fprintf(f, "\\u%04x", c);
        else fputc(c, f);
    }
    fputc('"', f);
}

static const char *label_to_prefix(const char *label) {
    if (strcmp(label, "private_person") == 0) return "PERSON";
    if (strcmp(label, "private_email") == 0) return "EMAIL";
    if (strcmp(label, "private_phone") == 0) return "PHONE";
    if (strcmp(label, "private_address") == 0) return "ADDRESS";
    if (strcmp(label, "private_date") == 0) return "DATE";
    if (strcmp(label, "account_number") == 0) return "ACCOUNT";
    if (strcmp(label, "private_url") == 0) return "URL";
    if (strcmp(label, "secret") == 0) return "SECRET";
    return "PII";
}

static int label_type_idx(const char *label) {
    static const char *t[] = {
        "account_number","private_address","private_date","private_email",
        "private_person","private_phone","private_url","secret"
    };
    for (int i = 0; i < 8; i++) if (strcmp(label, t[i]) == 0) return i;
    return -1;
}

static int run_mask(PFModel *model, const char *text) {
    PFSpan *spans = NULL;
    int count = 0;
    if (pf_predict(model, text, &spans, &count) != 0) return 1;

    int tc[8] = {0};
    char (*uids)[48] = (char (*)[48])calloc((size_t)(count > 0 ? count : 1), 48);
    if (!uids) { pf_free(spans); return 1; }

    for (int i = 0; i < count; i++) {
        int idx = label_type_idx(spans[i].label);
        int found = -1;
        uint32_t len_i = spans[i].end - spans[i].start;
        for (int j = 0; j < i; j++) {
            if (label_type_idx(spans[j].label) != idx) continue;
            uint32_t len_j = spans[j].end - spans[j].start;
            if (len_i == len_j && memcmp(text + spans[i].start, text + spans[j].start, len_i) == 0) {
                found = j; break;
            }
        }
        if (found >= 0) {
            memcpy(uids[i], uids[found], 48);
        } else {
            int n = (idx >= 0) ? ++tc[idx] : i + 1;
            snprintf(uids[i], 48, "%s_%d", label_to_prefix(spans[i].label), n);
        }
    }

    size_t text_len = strlen(text);
    size_t cap = text_len + (size_t)count * 32 + 1;
    char *masked = (char *)malloc(cap);
    if (!masked) { free(uids); pf_free(spans); return 1; }
    size_t w = 0;
    uint32_t cursor = 0;
    for (int i = 0; i < count; i++) {
        if (spans[i].start < cursor) continue;
        size_t n = spans[i].start - cursor;
        if (w + n + 64 > cap) {
            cap = (w + n + 64) * 2;
            char *tmp = (char *)realloc(masked, cap);
            if (!tmp) { free(masked); free(uids); pf_free(spans); return 1; }
            masked = tmp;
        }
        memcpy(masked + w, text + cursor, n);
        w += n;
        w += (size_t)snprintf(masked + w, cap - w, "<%s>", uids[i]);
        cursor = spans[i].end;
    }
    {
        size_t tail = text_len - cursor;
        if (w + tail + 1 > cap) {
            cap = w + tail + 1;
            char *tmp = (char *)realloc(masked, cap);
            if (!tmp) { free(masked); free(uids); pf_free(spans); return 1; }
            masked = tmp;
        }
        memcpy(masked + w, text + cursor, tail);
        w += tail;
    }
    masked[w] = 0;

    printf("{\n  \"masked_text\": ");
    fprint_json_str(stdout, masked, 0, (unsigned)w);
    printf(",\n  \"entities\": [");
    for (int i = 0; i < count; i++) {
        printf("\n    {\"uid\": \"%s\", \"label\": \"%s\", \"text\": ", uids[i], spans[i].label);
        fprint_json_str(stdout, text, spans[i].start, spans[i].end);
        printf("}%s", i + 1 < count ? "," : "");
    }
    printf("\n  ]\n}\n");

    free(masked);
    free(uids);
    pf_free(spans);
    return 0;
}

static char *json_extract_str(const char *json, size_t json_len, const char *key) {
    char needle[128];
    int nlen = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (nlen <= 0) return NULL;
    const char *end = json + json_len;
    const char *p = json;
    while ((p = (const char *)pf_memmem(p, (size_t)(end - p), needle, (size_t)nlen)) != NULL) {
        p += nlen;
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
        if (p < end && *p == ':') { p++; break; }
    }
    if (!p) return NULL;
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (p >= end || *p != '"') return NULL;
    p++;
    size_t cap = 256, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    while (p < end && *p != '"') {
        char c;
        if (*p == '\\' && p + 1 < end) {
            p++;
            switch (*p) {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case '/': c = '/'; break;
                default: c = *p; break;
            }
        } else {
            c = *p;
        }
        if (len + 2 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(out, cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }
        out[len++] = c;
        p++;
    }
    out[len] = 0;
    return out;
}


static char *str_replace_first(const char *haystack, const char *needle, const char *replacement) {
    const char *pos = strstr(haystack, needle);
    if (!pos) return NULL;
    size_t hlen = strlen(haystack);
    size_t nlen = strlen(needle);
    size_t rlen = strlen(replacement);
    char *out = (char *)malloc(hlen - nlen + rlen + 1);
    if (!out) return NULL;
    size_t prefix = (size_t)(pos - haystack);
    memcpy(out, haystack, prefix);
    memcpy(out + prefix, replacement, rlen);
    memcpy(out + prefix + rlen, pos + nlen, hlen - prefix - nlen + 1);
    return out;
}

static int run_unmask(const char *file_path) {
    char *json = file_path ? read_all_file(file_path) : read_all_stdin();
    if (!json) { fprintf(stderr, "failed to read stdin\n"); return 1; }
    size_t json_len = strlen(json);

    char *masked = json_extract_str(json, json_len, "masked_text");
    if (!masked) { fprintf(stderr, "missing \"masked_text\" in input\n"); free(json); return 1; }

    const char *ent = strstr(json, "\"entities\"");
    if (!ent) { fprintf(stderr, "missing \"entities\" in input\n"); free(masked); free(json); return 1; }

    const char *arr = strchr(ent, '[');
    if (!arr) { free(masked); free(json); return 1; }
    arr++;

    char *result = NULL;
    {
        size_t len = strlen(masked);
        result = (char *)malloc(len + 1);
        memcpy(result, masked, len + 1);
    }

    const char *p = arr;
    const char *json_end = json + json_len;
    while (p < json_end) {
        const char *obj = (const char *)memchr(p, '{', (size_t)(json_end - p));
        if (!obj) break;
        const char *obj_end = (const char *)memchr(obj, '}', (size_t)(json_end - obj));
        if (!obj_end) break;
        obj_end++;

        size_t olen = (size_t)(obj_end - obj);
        char *uid = json_extract_str(obj, olen, "uid");
        char *text = json_extract_str(obj, olen, "text");
        if (uid && text) {
            char placeholder[64];
            snprintf(placeholder, sizeof(placeholder), "<%s>", uid);
            char *next = str_replace_first(result, placeholder, text);
            if (next) { free(result); result = next; }
        }
        free(uid);
        free(text);
        p = obj_end;
    }

    fputs(result, stdout);
    fputc('\n', stdout);
    free(result);
    free(masked);
    free(json);
    return 0;
}

static void print_usage(FILE *out) {
    fputs("usage: privacy [--model DIR] [--file PATH] [--tokens|--redact|--mask|--unmask] [--] [text]\n", out);
    fputs("\nModes:\n", out);
    fputs("  (default)   Detect PII spans, output JSON array\n", out);
    fputs("  --redact    Replace PII with <REDACTED>\n", out);
    fputs("  --mask      Replace PII with <TYPE_N> UIDs, output JSON with mapping\n", out);
    fputs("  --unmask    Reverse --mask output (reads JSON from stdin or --file)\n", out);
    fputs("  --tokens    Show tokenization\n", out);
    fputs("\nInput: text argument, --file PATH, or stdin.\n\n", out);
    print_model_help(out);
}

enum { MODE_PREDICT, MODE_REDACT, MODE_MASK, MODE_UNMASK, MODE_TOKENS };

int main(int argc, char **argv) {
    const char *model_arg = NULL;
    const char *text = NULL;
    const char *file_path = NULL;
    int mode = MODE_PREDICT;
    int no_regex = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(stdout);
            return 0;
        } else if (strcmp(argv[i], "--version") == 0) {
            puts("privacy.cpp 0.1.0");
            return 0;
        } else if (strcmp(argv[i], "--") == 0) {
            if (i + 1 < argc) text = argv[++i];
            break;
        } else if (strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for --model\n"); return 2; }
            model_arg = argv[++i];
        } else if (strcmp(argv[i], "--file") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for --file\n"); return 2; }
            file_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0) {
            mode = MODE_TOKENS;
        } else if (strcmp(argv[i], "--redact") == 0) {
            mode = MODE_REDACT;
        } else if (strcmp(argv[i], "--mask") == 0) {
            mode = MODE_MASK;
        } else if (strcmp(argv[i], "--unmask") == 0) {
            mode = MODE_UNMASK;
        } else if (strcmp(argv[i], "--no-regex") == 0) {
            no_regex = 1;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            print_usage(stderr);
            return 2;
        } else {
            text = argv[i];
        }
    }

    if (mode == MODE_UNMASK) {
        return run_unmask(file_path);
    }

    char *file_text = NULL;
    if (file_path) {
        file_text = read_all_file(file_path);
        if (!file_text) return 1;
        text = file_text;
    }
    char *stdin_text = NULL;
    if (!text) {
        stdin_text = read_all_stdin();
        text = stdin_text;
    }
    if (!text) { print_usage(stderr); return 2; }

    char *model_dir = resolve_model_dir(model_arg, argc > 0 ? argv[0] : NULL);
    if (!model_dir) {
        fprintf(stderr, "no model directory found\n\n");
        print_model_help(stderr);
        free(stdin_text);
        free(file_text);
        return 2;
    }

    PFModel *model = pf_load_model(model_dir);
    if (!model) {
        fprintf(stderr, "failed to load model from %s\n", model_dir);
        free(model_dir);
        free(stdin_text);
        free(file_text);
        return 1;
    }

    int rc = 0;
    if (mode == MODE_TOKENS) {
        PFToken *tokens = NULL;
        int count = 0;
        if (pf_tokenize(model, text, &tokens, &count) != 0) { rc = 1; goto done; }
        printf("[\n");
        for (int i = 0; i < count; ++i) {
            printf("  {\"id\":%d,\"start\":%u,\"end\":%u,\"text\":", tokens[i].id, tokens[i].start, tokens[i].end);
            fprint_json_str(stdout, text, tokens[i].start, tokens[i].end);
            printf("}%s\n", i + 1 == count ? "" : ",");
        }
        printf("]\n");
        pf_free(tokens);
    } else if (mode == MODE_REDACT) {
        char *redacted = NULL;
        if (pf_redact(model, text, &redacted) != 0) { rc = 1; goto done; }
        puts(redacted);
        pf_free(redacted);
    } else if (mode == MODE_MASK) {
        rc = run_mask(model, text);
    } else {
        PFSpan *spans = NULL;
        int count = 0;
        int pred_rc = no_regex ? pf_predict_raw(model, text, &spans, &count)
                               : pf_predict(model, text, &spans, &count);
        if (pred_rc != 0) { rc = 1; goto done; }
        printf("[\n");
        for (int i = 0; i < count; ++i) {
            printf("  {\"label\":\"%s\",\"start\":%u,\"end\":%u,\"text\":", spans[i].label, spans[i].start, spans[i].end);
            fprint_json_str(stdout, text, spans[i].start, spans[i].end);
            printf("}%s\n", i + 1 == count ? "" : ",");
        }
        printf("]\n");
        pf_free(spans);
    }

done:
    pf_free_model(model);
    free(model_dir);
    free(stdin_text);
    free(file_text);
    return rc;
}
