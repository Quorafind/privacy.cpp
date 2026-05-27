# privacy.cpp

Local privacy redaction runtime in C, based on [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter). It loads the privacy-filter NER model and detects or redacts private spans such as names, emails, phone numbers, dates, addresses, URLs, account numbers, and secrets.

## Reference

This project is a C reimplementation of [OpenAI Privacy Filter](https://openai.com/index/open-sourcing-openai-privacy-filter/), a 1.5B-parameter (50M active) bidirectional token-classification model for PII detection and masking, released under the Apache 2.0 license.

- Model weights (Hugging Face): <https://huggingface.co/openai/privacy-filter>
- Interactive demo: <https://huggingface.co/spaces/openai/privacy-filter>
- Model card (PDF): <https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf>

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Windows Visual Studio generators usually produce:

```text
build\Release\privacy.exe
build\Release\privacy-bench.exe
```

Single-config generators usually produce:

```text
build/privacy
build/privacy-bench
```

Optional install step:

```sh
cmake --install build --config Release --prefix install
```

This installs the CLI tools under `install/bin` and the public C header under `install/include`.

## Model package format

A model package is a directory with fixed file names:

```text
privacy-ner-v1/
  tokenizer.json
  model.safetensors
  viterbi_calibration.json   # optional
```

Required files:

- `tokenizer.json`: Hugging Face style tokenizer JSON.
- `model.safetensors`: model weights.

Optional files:

- `viterbi_calibration.json`: decoding calibration data. If missing, the runtime uses built-in defaults.

When publishing a downloadable model, zip or tar the model directory itself:

```text
privacy-ner-v1.zip
  privacy-ner-v1/
    tokenizer.json
    model.safetensors
    viterbi_calibration.json
```

Keep the file names unchanged. Users should be able to unzip the package and point `privacy` at the extracted `privacy-ner-v1` directory.

## Install model

Recommended app layout:

```text
privacy.cpp/
  privacy.exe              # or privacy on Linux/macOS
  privacy-bench.exe
  models/
    privacy-ner-v1/
      tokenizer.json
      model.safetensors
      viterbi_calibration.json
```

With this layout, no `--model` argument is needed:

```sh
privacy --redact -- "Contact Alice at alice@example.com"
```

You can also install the model in any of these ways:

### 1. Pass the model path explicitly

```sh
privacy --model /path/to/privacy-ner-v1 --redact --file input.txt
```

### 2. Set an environment variable

```sh
# Linux/macOS
export PRIVACY_CPP_MODEL=/path/to/privacy-ner-v1

# Windows PowerShell
$env:PRIVACY_CPP_MODEL = "C:\\path\\to\\privacy-ner-v1"
```

Then run:

```sh
privacy --redact --file input.txt
```

`PRIVACY_FILTER_MODEL` is also supported for compatibility.

### 3. Install to the user cache directory

Windows:

```text
%LOCALAPPDATA%\privacy.cpp\models\privacy-ner-v1\
```

Linux/macOS:

```text
~/.cache/privacy.cpp/models/privacy-ner-v1/
```

## Model lookup order

If `--model` is not provided, `privacy` searches in this order:

1. `PRIVACY_CPP_MODEL`
2. `PRIVACY_FILTER_MODEL`
3. `./models/privacy-ner-v1`
4. `./model`
5. `<executable-dir>/models/privacy-ner-v1`
6. `<executable-dir>/model`
7. Windows: `%LOCALAPPDATA%\privacy.cpp\models\privacy-ner-v1`
8. Linux/macOS: `~/.cache/privacy.cpp/models/privacy-ner-v1`

## CLI usage

```text
usage: privacy [--model DIR] [--file PATH] [--tokens|--redact|--mask|--unmask] [--] [text]
```

Modes:

- default: detect private spans and output a JSON array.
- `--redact`: replace private spans with `<REDACTED>` and output plain text.
- `--mask`: replace private spans with stable placeholders such as `<EMAIL_1>` and output JSON with the mapping.
- `--unmask`: restore `--mask` JSON back to plain text.
- `--tokens`: output tokenizer result as JSON.

Other options:

- `--model DIR`: model directory.
- `--file PATH`: read input from a file.
- `--no-regex`: disable regex post-processing (model-only prediction).
- `--help`: print help.
- `--version`: print version.
- `--`: stop option parsing; useful when the input text starts with `-`.

## Input

`privacy` accepts input in three ways.

### 1. Text argument

```sh
privacy --redact -- "Contact Alice at alice@example.com"
```

### 2. File input

```sh
privacy --redact --file input.txt
```

### 3. Standard input

```sh
cat input.txt | privacy --redact
```

For `--unmask`, the input must be the JSON produced by `--mask`:

```sh
privacy --mask --file input.txt > masked.json
privacy --unmask --file masked.json
```

or:

```sh
cat masked.json | privacy --unmask
```

## Output

### Detect mode, default

Command:

```sh
privacy -- "Contact Alice at alice@example.com"
```

Output is a JSON array:

```json
[
  {"label":"private_person","start":8,"end":13,"text":"Alice"},
  {"label":"private_email","start":17,"end":34,"text":"alice@example.com"}
]
```

`start` and `end` are UTF-8 byte offsets into the original input. `end` is exclusive.

### Redact mode

Command:

```sh
privacy --redact -- "Contact Alice at alice@example.com"
```

Output is plain text:

```text
Contact <REDACTED> at <REDACTED>
```

### Mask mode

Command:

```sh
privacy --mask -- "Contact Alice at alice@example.com. Email Alice again."
```

Output is JSON:

```json
{
  "masked_text": "Contact <PERSON_1> at <EMAIL_1>. Email <PERSON_1> again.",
  "entities": [
    {"uid": "PERSON_1", "label": "private_person", "text": "Alice"},
    {"uid": "EMAIL_1", "label": "private_email", "text": "alice@example.com"},
    {"uid": "PERSON_1", "label": "private_person", "text": "Alice"}
  ]
}
```

Use this mode when another tool or model should process the masked text while preserving a reversible mapping.

### Unmask mode

Command:

```sh
privacy --unmask --file masked.json
```

Output is restored plain text:

```text
Contact Alice at alice@example.com. Email Alice again.
```

### Tokens mode

Command:

```sh
privacy --tokens -- "hello@example.com"
```

Output is a JSON array:

```json
[
  {"id":123,"start":0,"end":5,"text":"hello"}
]
```

The exact token IDs and splits depend on the model tokenizer.

## Common examples

Detect spans:

```sh
privacy --file input.txt
```

Redact a file:

```sh
privacy --redact --file input.txt > redacted.txt
```

Mask before sending text to another local tool:

```sh
privacy --mask --file input.txt > masked.json
```

Restore masked output:

```sh
privacy --unmask --file masked.json > restored.txt
```

Benchmark:

```sh
privacy-bench --model models/privacy-ner-v1 --repeat 3 --long 4000
```

## Benchmarks

Measured on Windows 10 x86-64, single-threaded (no OpenMP), no AVX2. Model: privacy-ner-v1 (2.7 GB safetensors). Binary size: ~90 KB.

### Throughput

| Input | Tokens | Predict (best of 5) | Tokens/sec |
|-------|-------:|--------------------:|-----------:|
| 1 sentence (56 B) | 16 | 31 ms | 516 |
| 500 words (3 KB) | 510 | 773 ms | 660 |
| 4000 words (24 KB) | 4010 | 4.75 s | 845 |
| long_test.txt (11 KB) | ~1500 | 8.1 s | ~185 |

Model load time: ~350 ms (memory-mapped weights).

### Accuracy

Evaluated on a synthetic multilingual test set (499 samples, 864 ground-truth spans). The test set includes English (60%), Chinese, Japanese, Korean, German, French, and Spanish text (35%), and API key/secret patterns (5%). Matching uses IoU >= 0.5 with label agreement.

Generate and evaluate:

```sh
python tools/eval_multilingual.py --binary build/Release/privacy.exe --model model --count 500
```

The detection pipeline combines the 1.5B neural model with a regex post-processing backstop. The model handles context-aware detection (names, addresses, contextual dates), while the regex backstop catches structurally-defined patterns the model may miss.

**Overall F1: 83.3% (model only) -> 87.8% (model+regex), +4.4%**

#### F1 by label: model vs model+regex

| Label | Model F1 | Model+Regex F1 | Delta |
|-------|:--------:|:--------------:|:-----:|
| account_number | 80.9% | 90.2% | +9.3% |
| private_address | 88.5% | 98.8% | +10.4% |
| private_date | 87.6% | 90.6% | +3.0% |
| private_email | 79.1% | 79.5% | +0.3% |
| private_person | 88.0% | 88.2% | +0.2% |
| private_phone | 77.6% | 85.4% | +7.8% |
| private_url | 13.3% | 100.0% | +86.7% |
| secret | 98.0% | 100.0% | +2.0% |

#### F1 by language: model vs model+regex

| Language | Model F1 | Model+Regex F1 | Delta |
|----------|:--------:|:--------------:|:-----:|
| English | 93.4% | 98.3% | +4.8% |
| Chinese | 24.6% | 43.8% | +19.2% |
| Japanese | 5.2% | 25.3% | +20.1% |
| Korean | 18.0% | 19.8% | +1.8% |
| German | 96.4% | 96.4% | = |
| French | 99.2% | 100.0% | +0.8% |
| Spanish | 100.0% | 100.0% | = |

The regex backstop has the largest impact on CJK languages (Chinese +19.2%, Japanese +20.1%) and URL detection (+86.7%), where the model's token-level classification struggles with non-Latin scripts and structurally-defined patterns.

### Regex backstop patterns

The regex post-processing layer detects:

| Pattern | Label | Examples |
|---------|-------|---------|
| International phone | `private_phone` | `+1-202-555-0147`, `+86 138 0013 8000`, `+33 1 23 45 67 89` |
| Parenthesized phone | `private_phone` | `(800) 555-0123` |
| Email | `private_email` | `user@example.com` |
| Long-form date | `private_date` | `January 15, 2024` |
| CJK date | `private_date` | `1990年5月27日` |
| European date | `private_date` | `15.03.2024`, `27/05/2024` |
| US SSN | `account_number` | `123-45-6789` |
| Chinese national ID | `account_number` | `110105199003078912`, `11010519900307891X` |
| Credit card | `account_number` | `4532 0151 2345 6789`, `5425-2334-3010-9903` |
| IBAN | `account_number` | `DE89 3704 0044 0532 0130 00` |
| EIN | `account_number` | `12-3456789` |
| IP address | `secret` | `192.168.1.1` |
| API key | `secret` | `sk-...`, `ghp_...`, `AKIA...`, `xoxb-...` |
| URL (20 schemes) | `private_url` | `https://`, `ssh://`, `mailto:`, `tel:` |
| Bare domain | `private_url` | 54 TLDs including `.cn`, `.jp`, `.kr`, `.de`, `.fr`, etc. |

### Multilingual support

Unit-tested languages:

| Language | Person | Email | Phone | Date | National ID | Account |
|----------|:------:|:-----:|:-----:|:----:|:-----------:|:-------:|
| English | model | regex | model+regex | model+regex | regex (SSN) | regex |
| Chinese (中文) | model | regex | regex (+86) | regex (年月日) | regex (18-digit) | - |
| Japanese (日本語) | model | regex | regex (+81) | regex (年月日) | - | - |
| Korean (한국어) | model | regex | regex (+82) | model | - | - |
| German (Deutsch) | model | regex | regex (+49) | regex (DD.MM.YYYY) | - | regex (IBAN) |
| French (Français) | model | regex | regex (+33) | regex (DD/MM/YYYY) | - | regex (IBAN) |
| Spanish (Español) | model | regex | regex (+34) | regex (DD.MM.YYYY) | - | regex (IBAN) |

Notes:
- URL detection supports 20 protocol schemes (`https`, `http`, `ftp`, `sftp`, `ssh`, `webdav`, `smb`, `nfs`, `ldap`, `vnc`, `rdp`, `s3`, `gs`, `mailto`, `tel`, etc.) and bare domain patterns with 54 TLDs.
- Email span tightening: model-detected email spans are refined to exact RFC-compliant boundaries, fixing imprecise boundaries in CJK contexts.
- Real-world accuracy may differ from synthetic benchmarks.

## C API

The public C API is in `src/privacy_filter.h`:

```c
PFModel *pf_load_model(const char *model_dir);
void pf_free_model(PFModel *model);
int pf_tokenize(PFModel *model, const char *text, PFToken **tokens_out, int *count_out);
int pf_predict(PFModel *model, const char *text, PFSpan **spans_out, int *count_out);
int pf_predict_raw(PFModel *model, const char *text, PFSpan **spans_out, int *count_out);
int pf_redact(PFModel *model, const char *text, char **redacted_out);
void pf_free(void *ptr);
```

`pf_predict` applies the full pipeline (model + regex backstop + email tightening). `pf_predict_raw` runs the model only, without regex post-processing.

Memory returned through `pf_tokenize`, `pf_predict`, `pf_predict_raw`, and `pf_redact` must be released with `pf_free`.

## Tests

Unit tests (includes multilingual test cases for Chinese, Japanese, Korean, German, French):

```sh
ctest --test-dir build -C Release --output-on-failure
```

Evaluate multilingual accuracy (model-only vs model+regex):

```sh
python tools/eval_multilingual.py --binary build/Release/privacy.exe --model model --count 500
```

The evaluator generates a multilingual dataset, runs both model-only and model+regex modes, and reports per-language and per-label F1 comparisons.

## Notes for model publishers

The runtime expects the architecture compiled into `src/privacy_filter.h` and a Hugging Face style `tokenizer.json` with BPE `ignore_merges=true`. The weights should be provided as `model.safetensors` with tensor names matching the runtime.

The current stable user-facing format is the model directory described above.
