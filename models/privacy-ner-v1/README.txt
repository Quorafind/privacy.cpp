Put model files in this directory:

  tokenizer.json
  model.safetensors
  viterbi_calibration.json    optional

After that, run from the project or release root:

  privacy --redact -- "Contact Alice at alice@example.com"

You can also pass a model directory explicitly:

  privacy --model models/privacy-ner-v1 --redact --file input.txt
