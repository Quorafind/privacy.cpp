#!/usr/bin/env python3
"""Generate a large-scale PII test set with ground truth annotations.

Produces a JSONL file where each line is:
    {"text": "...", "spans": [{"label": "...", "start": N, "end": N, "text": "..."}]}

Usage:
    python tools/generate_test_set.py [--out tests/bench_dataset.jsonl] [--count 500]
"""

import argparse
import json
import random
import string

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina", "Oscar", "Patricia",
    "Quincy", "Rachel", "Steven", "Tanya", "Uma", "Victor", "Wendy", "Xavier",
    "Yolanda", "Zachary", "Aisha", "Bjorn", "Catalina", "Dmitri", "Elena",
    "Farid", "Greta", "Hiroshi", "Ingrid", "Jamal", "Keiko", "Lorenzo",
    "Mei", "Nikolai", "Olga", "Pedro", "Qian", "Ravi", "Sakura", "Tariq",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
    "Martin", "Lee", "Clark", "Lewis", "Walker", "Hall", "Allen", "Young",
    "King", "Wright", "Torres", "Nguyen", "Hill", "Adams", "Baker", "Gonzalez",
    "Nelson", "Carter", "Mitchell", "Roberts", "Turner", "Phillips", "Campbell",
    "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris",
    "O'Brien", "Richardson", "Yamamoto", "Petrov", "Johansson", "Dubois",
]

DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "icloud.com",
    "hotmail.com", "fastmail.com", "aol.com", "mail.com", "zoho.com",
    "company.com", "corp.net", "enterprise.org", "startup.io", "firm.co",
]

STREETS = [
    "Main Street", "Oak Avenue", "Maple Drive", "Cedar Lane", "Pine Road",
    "Elm Street", "Washington Blvd", "Park Avenue", "Broadway", "Lake Street",
    "Highland Avenue", "Sunset Boulevard", "River Road", "Mountain View Dr",
    "Forest Lane", "Spring Street", "Harbor Boulevard", "Valley Road",
]

CITIES_STATES_ZIPS = [
    ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101"),
    ("Austin", "TX", "73301"), ("Jacksonville", "FL", "32099"),
    ("Columbus", "OH", "43085"), ("Charlotte", "NC", "28201"),
    ("Indianapolis", "IN", "46201"), ("Denver", "CO", "80201"),
    ("Seattle", "WA", "98101"), ("Boston", "MA", "02101"),
    ("Portland", "OR", "97201"), ("Nashville", "TN", "37201"),
]

# --- Multilingual names ---

CN_NAMES = [
    "张伟", "王芳", "李明", "赵丽", "刘洋", "陈静", "杨勇", "黄敏",
    "周磊", "吴桂英", "孙秀英", "马超", "朱军", "胡建华", "郭志强",
]

JP_NAMES = [
    "田中太郎", "山田花子", "佐藤健一", "鈴木美咲", "高橋大輔",
    "伊藤直子", "渡辺翔太", "中村由美", "小林一郎", "加藤さくら",
]

KR_NAMES = [
    "김민수", "이지연", "박준혁", "정유진", "최성호",
    "강미래", "조현우", "윤서연", "임태현", "한지민",
]

DE_FIRST = ["Hans", "Anna", "Klaus", "Petra", "Wolfgang", "Monika", "Stefan", "Ursula"]
DE_LAST = ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker"]

FR_FIRST = ["Marie", "Pierre", "Sophie", "Jean", "Isabelle", "Luc", "Claire", "Antoine"]
FR_LAST = ["Dupont", "Martin", "Bernard", "Dubois", "Moreau", "Laurent", "Simon", "Michel"]

ES_FIRST = ["Carlos", "Maria", "Juan", "Ana", "Luis", "Elena", "Miguel", "Carmen"]
ES_LAST = ["Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Hernandez", "Perez", "Sanchez"]

# --- Multilingual phone formats ---

INTL_PHONES = {
    "cn": lambda: f"+86 {random.randint(130,199)} {random.randint(1000,9999):04d} {random.randint(1000,9999):04d}",
    "jp": lambda: f"+81 {random.randint(3,99)}-{random.randint(1000,9999):04d}-{random.randint(1000,9999):04d}",
    "kr": lambda: f"+82 10-{random.randint(1000,9999):04d}-{random.randint(1000,9999):04d}",
    "de": lambda: f"+49 {random.randint(30,999)} {random.randint(10000000,99999999)}",
    "fr": lambda: f"+33 {random.randint(1,9)} {random.randint(10,99):02d} {random.randint(10,99):02d} {random.randint(10,99):02d} {random.randint(10,99):02d}",
    "uk": lambda: f"+44 {random.randint(20,79)} {random.randint(1000,9999):04d} {random.randint(1000,9999):04d}",
    "es": lambda: f"+34 {random.randint(600,699)} {random.randint(100,999):03d} {random.randint(100,999):03d}",
}

# --- Multilingual date formats ---

MONTHS_EN = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]


def random_date_cjk():
    y = random.randint(1950, 2024)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}年{m}月{d}日"


def random_date_euro_dot():
    y = random.randint(1950, 2024)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}.{m:02d}.{y}"


def random_date_euro_slash():
    y = random.randint(1950, 2024)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{d:02d}/{m:02d}/{y}"


def random_date_long():
    y = random.randint(1950, 2024)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{MONTHS_EN[m-1]} {d}, {y}"


# --- National IDs ---

def random_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def random_chinese_id():
    area = random.choice(["110105", "310101", "440305", "510107", "330102", "420111"])
    y = random.randint(1960, 2005)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    seq = random.randint(100, 999)
    base = f"{area}{y}{m:02d}{d:02d}{seq}"
    check = random.choice(list("0123456789X"))
    return base + check


# --- Credit cards ---

def random_credit_card():
    prefix = random.choice(["4", "5", "37", "6011"])
    total = 16 if prefix != "37" else 15
    rest = total - len(prefix)
    num = prefix + "".join(random.choices("0123456789", k=rest))
    sep = random.choice([" ", "-"])
    if total == 16:
        return f"{num[:4]}{sep}{num[4:8]}{sep}{num[8:12]}{sep}{num[12:16]}"
    return f"{num[:4]}{sep}{num[4:10]}{sep}{num[10:15]}"


# --- IBAN ---

IBAN_COUNTRIES = {
    "DE": 22, "GB": 22, "FR": 27, "ES": 24, "IT": 27, "NL": 18,
    "BE": 16, "AT": 20, "CH": 21, "SE": 24, "NO": 15, "DK": 18,
}


def random_iban():
    cc, length = random.choice(list(IBAN_COUNTRIES.items()))
    check = f"{random.randint(10,99)}"
    rest_len = length - 4
    rest = "".join(random.choices(string.ascii_uppercase + string.digits, k=rest_len))
    raw = cc + check + rest
    groups = [raw[i:i+4] for i in range(0, len(raw), 4)]
    return " ".join(groups)


# --- API keys / secrets ---

def random_api_key():
    kind = random.choice(["sk", "ghp", "akia", "xox"])
    if kind == "sk":
        body = "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(30, 50)))
        return f"sk-{body}"
    elif kind == "ghp":
        body = "".join(random.choices(string.ascii_letters + string.digits, k=36))
        return f"ghp_{body}"
    elif kind == "akia":
        body = "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        return f"AKIA{body}"
    else:
        body = "".join(random.choices(string.ascii_letters + string.digits + "-", k=30))
        return f"xoxb-{body}"


# --- Original generators ---

def random_person():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_email():
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower().replace("'", "")
    sep = random.choice([".", "_", ""])
    domain = random.choice(DOMAINS)
    return f"{first}{sep}{last}@{domain}"


def random_phone():
    formats = [
        lambda: f"({random.randint(200,999):03d}) {random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
        lambda: f"{random.randint(200,999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
        lambda: f"+1-{random.randint(200,999):03d}-{random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
        lambda: f"+1 ({random.randint(200,999):03d}) {random.randint(100,999):03d}-{random.randint(1000,9999):04d}",
    ]
    return random.choice(formats)()


def random_date():
    year = random.randint(1950, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    formats = [
        f"{year}-{month:02d}-{day:02d}",
        f"{month:02d}/{day:02d}/{year}",
        f"{MONTHS_EN[month-1]} {day}, {year}",
    ]
    return random.choice(formats)


def random_address():
    num = random.randint(1, 99999)
    street = random.choice(STREETS)
    city, state, zip_code = random.choice(CITIES_STATES_ZIPS)
    return f"{num} {street}, {city}, {state} {zip_code}"


def random_url():
    words = ["data", "api", "portal", "app", "dashboard", "cloud", "secure", "internal"]
    sub = random.choice(words)
    domain = random.choice(["example.com", "company.net", "service.org", "platform.io"])
    return f"https://{sub}.{domain}/path"


def random_account():
    kind = random.choice(["ein", "upper", "ssn", "cn_id", "credit_card", "iban"])
    if kind == "ein":
        return f"{random.randint(10,99)}-{random.randint(1000000,9999999)}"
    elif kind == "upper":
        letters = "".join(random.choices(string.ascii_uppercase, k=random.randint(6, 8)))
        digits = "".join(random.choices(string.digits, k=random.randint(2, 3)))
        return letters + digits
    elif kind == "ssn":
        return random_ssn()
    elif kind == "cn_id":
        return random_chinese_id()
    elif kind == "credit_card":
        return random_credit_card()
    else:
        return random_iban()


# --- Context templates ---

CONTEXT_TEMPLATES = [
    # Single entity templates
    ("Please contact {person}.", ["private_person"]),
    ("My name is {person} and I live here.", ["private_person"]),
    ("Send the invoice to {email}.", ["private_email"]),
    ("You can reach me at {email} anytime.", ["private_email"]),
    ("Call {phone} for more information.", ["private_phone"]),
    ("Her number is {phone}.", ["private_phone"]),
    ("We moved to {address}.", ["private_address"]),
    ("The office is at {address}.", ["private_address"]),
    ("Born on {date}, she graduated in 2020.", ["private_date"]),
    ("The meeting is scheduled for {date}.", ["private_date"]),
    ("Visit {url} for details.", ["private_url"]),
    ("Account number: {account}.", ["account_number"]),

    # Multi-entity templates
    ("Contact {person} at {email} or call {phone}.", ["private_person", "private_email", "private_phone"]),
    ("{person} lives at {address} and can be reached at {phone}.", ["private_person", "private_address", "private_phone"]),
    ("Dear {person}, your appointment on {date} has been confirmed. Please call {phone} if you need to reschedule.", ["private_person", "private_date", "private_phone"]),
    ("{person}'s email is {email}. Born {date}.", ["private_person", "private_email", "private_date"]),
    ("Patient {person}, DOB {date}, address {address}. Emergency contact: {phone}.", ["private_person", "private_date", "private_address", "private_phone"]),
    ("Employee {person} ({email}) joined on {date}. Office: {address}.", ["private_person", "private_email", "private_date", "private_address"]),

    # More naturalistic templates
    ("Hi, this is {person}. I wanted to follow up on the email I sent to {email} last week.", ["private_person", "private_email"]),
    ("The package was delivered to {address} on {date}.", ["private_address", "private_date"]),
    ("For billing inquiries, contact {person} at {phone} or email {email}.", ["private_person", "private_phone", "private_email"]),
    ("According to our records, {person} was born on {date} and resides at {address}.", ["private_person", "private_date", "private_address"]),

    # Negative / no-PII templates
    ("The weather today is sunny and warm.", []),
    ("Please review the quarterly report before the meeting.", []),
    ("The project deadline has been extended by two weeks.", []),
    ("We need to update the database schema for the new feature.", []),
    ("The server logs show increased traffic during peak hours.", []),
]

# --- Multilingual templates ---

CN_TEMPLATES = [
    ("联系 {person}，邮箱 {email}。", ["private_person", "private_email"]),
    ("电话 {phone}，请拨打。", ["private_phone"]),
    ("身份证号 {account}，已验证。", ["account_number"]),
    ("出生日期 {date}，已记录。", ["private_date"]),
    ("联系 {person}，邮箱 {email}，电话 {phone}。", ["private_person", "private_email", "private_phone"]),
]

JP_TEMPLATES = [
    ("{person}様、メール {email}。", ["private_person", "private_email"]),
    ("電話 {phone} です。", ["private_phone"]),
    ("誕生日 {date}。", ["private_date"]),
    ("{person}様、メール {email}、電話 {phone}。", ["private_person", "private_email", "private_phone"]),
]

KR_TEMPLATES = [
    ("{person}님, 이메일 {email}.", ["private_person", "private_email"]),
    ("전화 {phone} 입니다.", ["private_phone"]),
    ("{person}님, 이메일 {email}, 전화 {phone}.", ["private_person", "private_email", "private_phone"]),
]

DE_TEMPLATES = [
    ("Kontaktieren Sie {person} unter {email}.", ["private_person", "private_email"]),
    ("Tel. {phone}, geboren am {date}.", ["private_phone", "private_date"]),
    ("{person}, E-Mail {email}, IBAN {account}.", ["private_person", "private_email", "account_number"]),
]

FR_TEMPLATES = [
    ("Contactez {person} à {email}.", ["private_person", "private_email"]),
    ("Tél. {phone}, né(e) le {date}.", ["private_phone", "private_date"]),
    ("{person}, courriel {email}, tél. {phone}.", ["private_person", "private_email", "private_phone"]),
]

ES_TEMPLATES = [
    ("Contacte a {person} en {email}.", ["private_person", "private_email"]),
    ("Teléfono {phone}, nacido el {date}.", ["private_phone", "private_date"]),
    ("{person}, correo {email}, tel. {phone}.", ["private_person", "private_email", "private_phone"]),
]

# Secret templates
SECRET_TEMPLATES = [
    ("API key: {secret} stored in vault.", ["secret"]),
    ("Token {secret} expires in 30 days.", ["secret"]),
    ("Use {secret} to authenticate.", ["secret"]),
]


GENERATORS = {
    "private_person": random_person,
    "private_email": random_email,
    "private_phone": random_phone,
    "private_date": random_date,
    "private_address": random_address,
    "private_url": random_url,
    "account_number": random_account,
}

PLACEHOLDER_MAP = {
    "private_person": "{person}",
    "private_email": "{email}",
    "private_phone": "{phone}",
    "private_date": "{date}",
    "private_address": "{address}",
    "private_url": "{url}",
    "account_number": "{account}",
    "secret": "{secret}",
}


def gen_cn_person():
    return random.choice(CN_NAMES)

def gen_jp_person():
    return random.choice(JP_NAMES)

def gen_kr_person():
    return random.choice(KR_NAMES)

def gen_de_person():
    return f"{random.choice(DE_FIRST)} {random.choice(DE_LAST)}"

def gen_fr_person():
    return f"{random.choice(FR_FIRST)} {random.choice(FR_LAST)}"

def gen_es_person():
    return f"{random.choice(ES_FIRST)} {random.choice(ES_LAST)}"


def gen_intl_email(lang):
    name_gens = {
        "cn": lambda: random.choice(CN_NAMES),
        "jp": lambda: random.choice(JP_NAMES),
        "kr": lambda: random.choice(KR_NAMES),
        "de": gen_de_person,
        "fr": gen_fr_person,
        "es": gen_es_person,
    }
    tlds = {"cn": ".cn", "jp": ".jp", "kr": ".kr", "de": ".de", "fr": ".fr", "es": ".es"}
    domain = random.choice(["example", "company", "mail"]) + tlds.get(lang, ".com")
    # Use ASCII-safe local part
    first = random.choice(FIRST_NAMES).lower()
    last = random.choice(LAST_NAMES).lower().replace("'", "")
    return f"{first}.{last}@{domain}"


INTL_GENERATORS = {
    "cn": {
        "private_person": gen_cn_person,
        "private_email": lambda: gen_intl_email("cn"),
        "private_phone": INTL_PHONES["cn"],
        "private_date": random_date_cjk,
        "account_number": random_chinese_id,
    },
    "jp": {
        "private_person": gen_jp_person,
        "private_email": lambda: gen_intl_email("jp"),
        "private_phone": INTL_PHONES["jp"],
        "private_date": random_date_cjk,
        "account_number": random_account,
    },
    "kr": {
        "private_person": gen_kr_person,
        "private_email": lambda: gen_intl_email("kr"),
        "private_phone": INTL_PHONES["kr"],
        "private_date": random_date,
        "account_number": random_account,
    },
    "de": {
        "private_person": gen_de_person,
        "private_email": lambda: gen_intl_email("de"),
        "private_phone": INTL_PHONES["de"],
        "private_date": random_date_euro_dot,
        "account_number": random_iban,
    },
    "fr": {
        "private_person": gen_fr_person,
        "private_email": lambda: gen_intl_email("fr"),
        "private_phone": INTL_PHONES["fr"],
        "private_date": random_date_euro_slash,
        "account_number": random_iban,
    },
    "es": {
        "private_person": gen_es_person,
        "private_email": lambda: gen_intl_email("es"),
        "private_phone": INTL_PHONES["es"],
        "private_date": random_date_euro_dot,
        "account_number": random_iban,
    },
}

INTL_TEMPLATE_GROUPS = {
    "cn": CN_TEMPLATES,
    "jp": JP_TEMPLATES,
    "kr": KR_TEMPLATES,
    "de": DE_TEMPLATES,
    "fr": FR_TEMPLATES,
    "es": ES_TEMPLATES,
}


def char_to_byte_offset(text, char_idx):
    return len(text[:char_idx].encode("utf-8"))


def generate_sample(template, labels, generators=None):
    if generators is None:
        generators = GENERATORS
    text = template
    spans = []
    for label in labels:
        placeholder = PLACEHOLDER_MAP[label]
        if label in generators:
            value = generators[label]()
        elif label == "secret":
            value = random_api_key()
        else:
            value = GENERATORS[label]()
        idx = text.find(placeholder)
        if idx < 0:
            continue
        text = text[:idx] + value + text[idx + len(placeholder):]
        byte_start = char_to_byte_offset(text, idx)
        byte_end = byte_start + len(value.encode("utf-8"))
        spans.append({
            "label": label,
            "start": byte_start,
            "end": byte_end,
            "text": value,
        })
    return {"text": text, "spans": spans}


LANG_DISPLAY = {
    "en": "English", "cn": "Chinese", "jp": "Japanese",
    "kr": "Korean", "de": "German", "fr": "French", "es": "Spanish",
}


def generate_dataset(count, seed=42):
    random.seed(seed)
    samples = []

    # 60% English templates
    en_count = int(count * 0.60)
    for _ in range(en_count):
        template, labels = random.choice(CONTEXT_TEMPLATES)
        sample = generate_sample(template, labels)
        sample["lang"] = "en"
        samples.append(sample)

    # 5% secret templates
    secret_count = int(count * 0.05)
    for _ in range(secret_count):
        template, labels = random.choice(SECRET_TEMPLATES)
        sample = generate_sample(template, labels)
        sample["lang"] = "en"
        samples.append(sample)

    # 35% multilingual templates (split across languages)
    intl_count = count - en_count - secret_count
    langs = list(INTL_TEMPLATE_GROUPS.keys())
    per_lang = max(1, intl_count // len(langs))
    for lang in langs:
        templates = INTL_TEMPLATE_GROUPS[lang]
        gens = INTL_GENERATORS[lang]
        for _ in range(per_lang):
            template, labels = random.choice(templates)
            sample = generate_sample(template, labels, gens)
            sample["lang"] = lang
            samples.append(sample)

    random.shuffle(samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate PII test dataset")
    parser.add_argument("--out", default="tests/bench_dataset.jsonl")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = generate_dataset(args.count, args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    total_spans = sum(len(s["spans"]) for s in samples)
    label_counts = {}
    lang_counts = {}
    for s in samples:
        for sp in s["spans"]:
            label_counts[sp["label"]] = label_counts.get(sp["label"], 0) + 1
        text = s["text"]
        if any(ord(c) > 0x3000 for c in text):
            if any("一" <= c <= "鿿" for c in text):
                lang_counts["CJK"] = lang_counts.get("CJK", 0) + 1
            elif any("가" <= c <= "힯" for c in text):
                lang_counts["Korean"] = lang_counts.get("Korean", 0) + 1
        elif any(c in text for c in "àéüöäñ"):
            lang_counts["European"] = lang_counts.get("European", 0) + 1
        else:
            lang_counts["English"] = lang_counts.get("English", 0) + 1

    print(f"Generated {len(samples)} samples with {total_spans} total PII spans")
    print(f"Output: {args.out}")
    print("Label distribution:")
    for label, cnt in sorted(label_counts.items()):
        print(f"  {label}: {cnt}")
    print("Language distribution:")
    for lang, cnt in sorted(lang_counts.items()):
        print(f"  {lang}: {cnt}")


if __name__ == "__main__":
    main()
