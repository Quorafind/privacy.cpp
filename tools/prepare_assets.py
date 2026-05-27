import json
import os
import struct
import sys
from pathlib import Path

VOCAB_MAGIC = b"PFVOCAB1"
INDEX_MAGIC = b"PFIDX001"


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def token_to_bytes(token, inverse):
    out = bytearray()
    for ch in token:
        try:
            out.append(inverse[ch])
        except KeyError as exc:
            raise ValueError(f"token contains non byte-level character: {token!r} char={ch!r}") from exc
    return bytes(out)


def write_vocab(model_dir: Path):
    tokenizer_path = model_dir / "tokenizer.json"
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    model = tokenizer.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError("Only BPE tokenizer.json is supported")
    if model.get("ignore_merges") is not True:
        raise ValueError("This converter expects tokenizer model.ignore_merges=true")

    inverse = {v: k for k, v in bytes_to_unicode().items()}
    entries = []
    for token, token_id in model["vocab"].items():
        entries.append((int(token_id), token_to_bytes(token, inverse)))
    entries.sort(key=lambda item: item[0])

    specials = []
    for item in tokenizer.get("added_tokens", []):
        if item.get("special"):
            specials.append((int(item["id"]), item["content"].encode("utf-8")))
    specials.sort(key=lambda item: item[0])

    out_path = model_dir / "pf_vocab.bin"
    with out_path.open("wb") as f:
        f.write(VOCAB_MAGIC)
        f.write(struct.pack("<II", len(entries), len(specials)))
        for token_id, raw in entries:
            f.write(struct.pack("<II", token_id, len(raw)))
            f.write(raw)
        for token_id, raw in specials:
            f.write(struct.pack("<II", token_id, len(raw)))
            f.write(raw)
    print(f"wrote {out_path} ({len(entries)} vocab entries, {len(specials)} specials)")


def write_safetensors_index(model_dir: Path):
    st_path = model_dir / "model.safetensors"
    with st_path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("Invalid safetensors file")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header = json.loads(f.read(header_len))

    data_start = 8 + header_len
    tensors = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = [int(x) for x in info["shape"]]
        offsets = [int(x) for x in info["data_offsets"]]
        if dtype == "BF16":
            dtype_code = 1
        elif dtype == "F32":
            dtype_code = 2
        else:
            raise ValueError(f"Unsupported dtype for {name}: {dtype}")
        if len(shape) > 4:
            raise ValueError(f"Unsupported rank for {name}: {shape}")
        dims = shape + [1] * (4 - len(shape))
        nbytes = offsets[1] - offsets[0]
        tensors.append((name, dtype_code, len(shape), dims, data_start + offsets[0], nbytes))
    tensors.sort(key=lambda item: item[0])

    out_path = model_dir / "pf_weights.idx"
    with out_path.open("wb") as f:
        f.write(INDEX_MAGIC)
        f.write(struct.pack("<I", len(tensors)))
        for name, dtype_code, ndim, dims, offset, nbytes in tensors:
            raw_name = name.encode("utf-8")
            if len(raw_name) > 127:
                raise ValueError(f"Tensor name too long: {name}")
            f.write(struct.pack("<H", len(raw_name)))
            f.write(raw_name)
            f.write(struct.pack("<II4QQQ", dtype_code, ndim, dims[0], dims[1], dims[2], dims[3], offset, nbytes))
    print(f"wrote {out_path} ({len(tensors)} tensors, data file {st_path.name})")


def main():
    model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    write_vocab(model_dir)
    write_safetensors_index(model_dir)


if __name__ == "__main__":
    main()
