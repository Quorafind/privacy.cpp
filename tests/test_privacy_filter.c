#include "privacy_filter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect_true(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static int slice_equals(const char *text, uint32_t start, uint32_t end, const char *want) {
    size_t n = strlen(want);
    return end >= start && (size_t)(end - start) == n && memcmp(text + start, want, n) == 0;
}

static int has_span(PFSpan *spans, int count, const char *text, const char *label, const char *value) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(spans[i].label, label) == 0 && slice_equals(text, spans[i].start, spans[i].end, value)) return 1;
    }
    return 0;
}

static void test_tokenizer(PFModel *model) {
    const char *text = "Alice was born on 1990-01-02.";
    PFToken *tokens = NULL;
    int count = 0;
    expect_true(pf_tokenize(model, text, &tokens, &count) == 0, "tokenizer returns success");
    expect_true(count > 0, "tokenizer emits tokens");
    for (int i = 0; i < count; ++i) {
        expect_true(tokens[i].end > tokens[i].start, "token has positive byte range");
        if (i > 0) expect_true(tokens[i - 1].end == tokens[i].start, "tokens are contiguous");
    }
    pf_free(tokens);
}

static void test_long_tokenizer(PFModel *model) {
    char text[4096];
    size_t pos = 0;
    for (int i = 0; i < 700 && pos + 6 < sizeof(text); ++i) {
        memcpy(text + pos, "hello ", 6);
        pos += 6;
    }
    text[pos] = 0;

    PFToken *tokens = NULL;
    int count = 0;
    expect_true(pf_tokenize(model, text, &tokens, &count) == 0, "long tokenizer returns success");
    expect_true(count > 512, "long tokenizer supports more than 512 tokens");
    pf_free(tokens);
}

static void test_date(PFModel *model) {
    const char *text = "Alice was born on 1990-01-02.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "date prediction returns success");
    expect_true(has_span(spans, count, text, "private_date", "1990-01-02"), "detects ISO date");
    pf_free(spans);
}

static void test_contact(PFModel *model) {
    const char *text = "Contact Alice at alice@example.com or call 555-123-4567.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "contact prediction returns success");
    expect_true(has_span(spans, count, text, "private_person", "Alice"), "detects person");
    expect_true(has_span(spans, count, text, "private_email", "alice@example.com"), "detects email");
    expect_true(has_span(spans, count, text, "private_phone", "555-123-4567"), "detects phone");
    pf_free(spans);
}

static void test_url(PFModel *model) {
    const char *text = "Visit https://portal.example.com/dashboard for details.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "url prediction returns success");
    expect_true(has_span(spans, count, text, "private_url", "https://portal.example.com/dashboard"), "detects https url");
    pf_free(spans);
}

static void test_url_protocols(PFModel *model) {
    {
        const char *text = "Connect via ssh://admin@server.local:22 or webdav://files.corp.net/docs please.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "protocol url prediction returns success");
        expect_true(has_span(spans, count, text, "private_url", "ssh://admin@server.local:22"), "detects ssh url");
        expect_true(has_span(spans, count, text, "private_url", "webdav://files.corp.net/docs"), "detects webdav url");
        pf_free(spans);
    }
    {
        const char *text = "Upload to s3://my-bucket/data.csv now.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "s3 url prediction returns success");
        expect_true(has_span(spans, count, text, "secret", "s3://my-bucket/data.csv"), "detects s3 url as secret");
        pf_free(spans);
    }
}

/* ---- Multilingual & extended pattern tests ---- */

static void test_intl_phones(PFModel *model) {
    {
        const char *text = "Call +1-202-555-0147 for US support.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone US returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+1-202-555-0147"), "detects +1 US phone");
        pf_free(spans);
    }
    {
        const char *text = "Reach us at +1 (415) 555-0198 anytime.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone US paren returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+1 (415) 555-0198"), "detects +1 (xxx) phone");
        pf_free(spans);
    }
    {
        const char *text = "Call (800) 555-0123 for info.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "paren phone returns success");
        expect_true(has_span(spans, count, text, "private_phone", "(800) 555-0123"), "detects (xxx) xxx-xxxx phone");
        pf_free(spans);
    }
    {
        const char *text = "UK office: +44 20 7946 0958 available.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone UK returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+44 20 7946 0958"), "detects UK phone");
        pf_free(spans);
    }
    {
        const char *text = "Germany: +49 30 901820 here.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone DE returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+49 30 901820"), "detects DE phone");
        pf_free(spans);
    }
    {
        /* +86 138 0013 8000 — Chinese mobile */
        const char *text = "\xe8\x81\x94\xe7\xb3\xbb +86 138 0013 8000 \xe8\xaf\xb7\xe6\x8b\xa8\xe6\x89\x93\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone CN returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+86 138 0013 8000"), "detects CN phone");
        pf_free(spans);
    }
    {
        /* +81 3-1234-5678 — Japanese landline */
        const char *text = "\xe9\x80\xa3\xe7\xb5\xa1\xe5\x85\x88 +81 3-1234-5678 \xe3\x81\xa7\xe3\x81\x99\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone JP returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+81 3-1234-5678"), "detects JP phone");
        pf_free(spans);
    }
    {
        /* +82 10-1234-5678 — Korean mobile */
        const char *text = "\xec\xa0\x84\xed\x99\x94 +82 10-1234-5678 \xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone KR returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+82 10-1234-5678"), "detects KR phone");
        pf_free(spans);
    }
    {
        /* +33 1 23 45 67 89 — French */
        const char *text = "Appelez +33 1 23 45 67 89 pour aide.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "intl phone FR returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+33 1 23 45 67 89"), "detects FR phone");
        pf_free(spans);
    }
}

static void test_date_formats(PFModel *model) {
    {
        const char *text = "Born on January 15, 2024 at the hospital.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "long date returns success");
        expect_true(has_span(spans, count, text, "private_date", "January 15, 2024"), "detects long-form English date");
        pf_free(spans);
    }
    {
        const char *text = "Event on September 3, 1985 was recorded.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "long date Sep returns success");
        expect_true(has_span(spans, count, text, "private_date", "September 3, 1985"), "detects September date");
        pf_free(spans);
    }
    {
        /* 15.03.2024 — European dot format */
        const char *text = "Geburtsdatum: 15.03.2024 bitte notieren.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "euro date dot returns success");
        expect_true(has_span(spans, count, text, "private_date", "15.03.2024"), "detects DD.MM.YYYY date");
        pf_free(spans);
    }
    {
        /* 27/05/2024 — European slash format */
        const char *text = "Date de naissance: 27/05/2024 enregistr\xc3\xa9.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "euro date slash returns success");
        expect_true(has_span(spans, count, text, "private_date", "27/05/2024"), "detects DD/MM/YYYY date");
        pf_free(spans);
    }
    {
        /* 1990\xe5\xb9\xb45\xe6\x9c\x8827\xe6\x97\xa5 — CJK: 1990年5月27日 */
        const char *text = "\xe5\x87\xba\xe7\x94\x9f\xe6\x97\xa5\xe6\x98\xaf 1990\xe5\xb9\xb4"
                           "5\xe6\x9c\x88" "27\xe6\x97\xa5\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "CJK date returns success");
        expect_true(has_span(spans, count, text, "private_date",
                             "1990\xe5\xb9\xb4" "5\xe6\x9c\x88" "27\xe6\x97\xa5"), "detects CJK date YYYY-M-D");
        pf_free(spans);
    }
}

static void test_national_ids(PFModel *model) {
    {
        /* US SSN */
        const char *text = "SSN is 123-45-6789 on file.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "SSN returns success");
        expect_true(has_span(spans, count, text, "account_number", "123-45-6789"), "detects US SSN");
        pf_free(spans);
    }
    {
        /* Chinese ID: 18 digits, last may be X */
        const char *text = "\xe8\xba\xab\xe4\xbb\xbd\xe8\xaf\x81\xe5\x8f\xb7 11010519900307891X \xe5\xb7\xb2\xe9\xaa\x8c\xe8\xaf\x81\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "Chinese ID returns success");
        expect_true(has_span(spans, count, text, "account_number", "11010519900307891X"), "detects Chinese national ID");
        pf_free(spans);
    }
}

static void test_credit_cards(PFModel *model) {
    {
        const char *text = "Card: 4532 0151 2345 6789 on record.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "credit card returns success");
        expect_true(has_span(spans, count, text, "account_number", "4532 0151 2345 6789"), "detects credit card with spaces");
        pf_free(spans);
    }
    {
        const char *text = "Charged 5425-2334-3010-9903 today.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "credit card dash returns success");
        expect_true(has_span(spans, count, text, "account_number", "5425-2334-3010-9903"), "detects credit card with dashes");
        pf_free(spans);
    }
}

static void test_iban(PFModel *model) {
    {
        const char *text = "Transfer to DE89 3704 0044 0532 0130 00 today.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "IBAN returns success");
        expect_true(has_span(spans, count, text, "account_number", "DE89 3704 0044 0532 0130 00"), "detects German IBAN");
        pf_free(spans);
    }
    {
        const char *text = "Send to GB29 NWBK 6016 1331 9268 19 please.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "UK IBAN returns success");
        expect_true(has_span(spans, count, text, "account_number", "GB29 NWBK 6016 1331 9268 19"), "detects UK IBAN");
        pf_free(spans);
    }
}

static void test_api_keys(PFModel *model) {
    {
        const char *text = "Key: sk-proj-abc123def456ghi789jkl012mno345pqr678stu901 stored.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "API key sk- returns success");
        expect_true(has_span(spans, count, text, "secret", "sk-proj-abc123def456ghi789jkl012mno345pqr678stu901"), "detects sk- API key");
        pf_free(spans);
    }
    {
        const char *text = "Token: ghp_1234567890abcdefghijABCDEFGHIJ1234567890 saved.";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "GitHub token returns success");
        expect_true(has_span(spans, count, text, "secret", "ghp_1234567890abcdefghijABCDEFGHIJ1234567890"), "detects ghp_ GitHub token");
        pf_free(spans);
    }
}

static void test_chinese_text(PFModel *model) {
    {
        /* Email in Chinese context: 邮箱 zhangwei@example.cn */
        const char *text = "\xe9\x82\xae\xe7\xae\xb1 zhangwei@example.cn \xe8\xaf\xb7\xe6\x9f\xa5\xe6\x94\xb6\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "CN email returns success");
        expect_true(has_span(spans, count, text, "private_email", "zhangwei@example.cn"), "detects email in Chinese text");
        pf_free(spans);
    }
    {
        /* Phone in Chinese context: 电话 +86 139 0001 2345 */
        const char *text = "\xe7\x94\xb5\xe8\xaf\x9d +86 139 0001 2345\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "CN phone returns success");
        expect_true(has_span(spans, count, text, "private_phone", "+86 139 0001 2345"), "detects CN phone in Chinese text");
        pf_free(spans);
    }
    {
        /* Chinese national ID: 身份证号 110105199003078912 */
        const char *text = "\xe8\xba\xab\xe4\xbb\xbd\xe8\xaf\x81\xe5\x8f\xb7 110105199003078912 \xe5\xb7\xb2\xe9\xaa\x8c\xe8\xaf\x81\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "CN ID returns success");
        expect_true(has_span(spans, count, text, "account_number", "110105199003078912"), "detects Chinese ID in Chinese text");
        pf_free(spans);
    }
    {
        /* CJK date: 出生日期 1990年3月7日 */
        const char *text = "\xe5\x87\xba\xe7\x94\x9f\xe6\x97\xa5\xe6\x9c\x9f 1990\xe5\xb9\xb4"
                           "3\xe6\x9c\x88" "7\xe6\x97\xa5\xe3\x80\x82";
        PFSpan *spans = NULL;
        int count = 0;
        expect_true(pf_predict(model, text, &spans, &count) == 0, "CJK date CN returns success");
        expect_true(has_span(spans, count, text, "private_date",
                             "1990\xe5\xb9\xb4" "3\xe6\x9c\x88" "7\xe6\x97\xa5"), "detects CJK date in Chinese text");
        pf_free(spans);
    }
}

static void test_japanese_text(PFModel *model) {
    /* 田中太郎様、メール tanaka@example.jp、電話 +81 3-5555-1234。 */
    const char *text =
        "\xe7\x94\xb0\xe4\xb8\xad\xe5\xa4\xaa\xe9\x83\x8e\xe6\xa7\x98\xe3\x80\x81"
        "\xe3\x83\xa1\xe3\x83\xbc\xe3\x83\xab tanaka@example.jp\xe3\x80\x81"
        "\xe9\x9b\xbb\xe8\xa9\xb1 +81 3-5555-1234\xe3\x80\x82";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "Japanese text returns success");
    expect_true(has_span(spans, count, text, "private_email", "tanaka@example.jp"), "detects email in Japanese text");
    expect_true(has_span(spans, count, text, "private_phone", "+81 3-5555-1234"), "detects JP phone in Japanese text");
    pf_free(spans);
}

static void test_korean_text(PFModel *model) {
    /* 김민수님, 이메일 kim@example.kr, 전화 +82 10-9876-5432. */
    const char *text =
        "\xea\xb9\x80\xeb\xaf\xbc\xec\x88\x98\xeb\x8b\x98, "
        "\xec\x9d\xb4\xeb\xa9\x94\xec\x9d\xbc kim@example.kr, "
        "\xec\xa0\x84\xed\x99\x94 +82 10-9876-5432.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "Korean text returns success");
    expect_true(has_span(spans, count, text, "private_email", "kim@example.kr"), "detects email in Korean text");
    expect_true(has_span(spans, count, text, "private_phone", "+82 10-9876-5432"), "detects KR phone in Korean text");
    pf_free(spans);
}

static void test_german_text(PFModel *model) {
    /* Kontaktieren Sie Hans Mueller unter hans.mueller@example.de,
       Tel. +49 30 12345678, geboren am 15.03.1985,
       IBAN DE89 3704 0044 0532 0130 00. */
    const char *text =
        "Kontaktieren Sie Hans Mueller unter hans.mueller@example.de, "
        "Tel. +49 30 12345678, geboren am 15.03.1985, "
        "IBAN DE89 3704 0044 0532 0130 00.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "German text returns success");
    expect_true(has_span(spans, count, text, "private_email", "hans.mueller@example.de"), "detects email in German text");
    expect_true(has_span(spans, count, text, "private_phone", "+49 30 12345678"), "detects DE phone in German text");
    expect_true(has_span(spans, count, text, "private_date", "15.03.1985"), "detects European date in German text");
    {
        int iban_found = has_span(spans, count, text, "account_number", "DE89 3704 0044 0532 0130 00")
                      || has_span(spans, count, text, "account_number", "IBAN DE89 3704 0044 0532 0130 00");
        expect_true(iban_found, "detects IBAN in German text");
    }
    pf_free(spans);
}

static void test_french_text(PFModel *model) {
    /* Contactez Marie Dupont \xc3\xa0 marie.dupont@example.fr,
       t\xc3\xa9l. +33 1 23 45 67 89, n\xc3\xa9e le 27/05/1992. */
    const char *text =
        "Contactez Marie Dupont \xc3\xa0 marie.dupont@example.fr, "
        "t\xc3\xa9l. +33 1 23 45 67 89, n\xc3\xa9\x65 le 27/05/1992.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "French text returns success");
    expect_true(has_span(spans, count, text, "private_email", "marie.dupont@example.fr"), "detects email in French text");
    expect_true(has_span(spans, count, text, "private_phone", "+33 1 23 45 67 89"), "detects FR phone in French text");
    expect_true(has_span(spans, count, text, "private_date", "27/05/1992"), "detects European date in French text");
    pf_free(spans);
}

static void test_mixed_multilingual(PFModel *model) {
    /* Mix of English and Chinese with multiple PII types */
    const char *text =
        "Customer \xe5\xbc\xa0\xe4\xbc\x9f (Zhang Wei) email: zhang.wei@corp.cn, "
        "card 4111 1111 1111 1111, DOB 1988\xe5\xb9\xb4"
        "12\xe6\x9c\x88" "25\xe6\x97\xa5.";
    PFSpan *spans = NULL;
    int count = 0;
    expect_true(pf_predict(model, text, &spans, &count) == 0, "mixed multilingual returns success");
    expect_true(has_span(spans, count, text, "private_email", "zhang.wei@corp.cn"), "detects email in mixed text");
    expect_true(has_span(spans, count, text, "account_number", "4111 1111 1111 1111"), "detects credit card in mixed text");
    expect_true(has_span(spans, count, text, "private_date",
                         "1988\xe5\xb9\xb4" "12\xe6\x9c\x88" "25\xe6\x97\xa5"), "detects CJK date in mixed text");
    pf_free(spans);
}

int main(int argc, char **argv) {
    const char *model_dir = "model";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
    }

    PFModel *model = pf_load_model(model_dir);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model from %s\n", model_dir);
        return 1;
    }

    test_tokenizer(model);
    test_long_tokenizer(model);
    test_date(model);
    test_contact(model);
    test_url(model);
    test_url_protocols(model);

    /* Multilingual & extended pattern tests */
    test_intl_phones(model);
    test_date_formats(model);
    test_national_ids(model);
    test_credit_cards(model);
    test_iban(model);
    test_api_keys(model);
    test_chinese_text(model);
    test_japanese_text(model);
    test_korean_text(model);
    test_german_text(model);
    test_french_text(model);
    test_mixed_multilingual(model);

    pf_free_model(model);
    if (failures) {
        fprintf(stderr, "%d test failure(s)\n", failures);
        return 1;
    }
    puts("privacy_filter_tests: ok");
    return 0;
}
