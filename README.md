# Fuel Receipt OCR · Structured Field Extraction

Extract structured fields from fuel receipt images using a fine-tuned QA model. Fully local, no cloud API, runs on CPU.

---

## What it does

Takes a fuel receipt image → runs OCR → asks natural-language questions against the extracted text → returns structured JSON.

```json
{
  "station_name": { "value": "Shell India Pump",        "confidence": 0.94 },
  "station_addr": { "value": "251 High Street, Mumbai", "confidence": 0.87 },
  "date":         { "value": "15/08/2023 10:30",        "confidence": 0.96 },
  "fuel_type":    { "value": "Regular Unleaded",        "confidence": 0.91 },
  "unit_price":   { "value": "102.45",                  "confidence": 0.89 },
  "volume":       { "value": "25.500",                  "confidence": 0.92 },
  "total_amount": { "value": "2612.47",                 "confidence": 0.93 },
  "currency":     { "value": "₹",                       "confidence": 0.88 }
}
```

---

## Architecture

```
Receipt Image
     │
     ▼
Tesseract OCR  ────────────────── text lines (pytesseract)
     │
     ▼
Sliding Window Context  ──────── 384-token windows, 128-token stride
     │
     ▼
QA Model (BERT-based) ─────────  "What is the total amount paid?" → "2612.47"
     │
     ▼
Structured JSON Output
```

**Two deployment modes:**

| Mode | Model | Dependencies | Use case |
|------|-------|-------------|----------|
| Dev | Transformers pipeline | `transformers` + `torch` | Evaluation, fine-tuning iterations |
| Prod | ONNX Runtime + custom tokenizer | `onnxruntime` + `pytesseract` | Embedded, edge, minimal footprint |

---

## Model

Base: [`deepset/minilm-uncased-squad2`](https://huggingface.co/deepset/minilm-uncased-squad2)

Architecture: MiniLM L12-H384 (~33M parameters), fine-tuned on SQuAD2, then fine-tuned here on synthetic fuel receipt data.

| Format | Size | Notes |
|--------|------|-------|
| `model_fp32.onnx` | ~133 MB | Full precision ONNX |
| `model_int8.onnx` | ~33 MB | INT8 dynamic quantization, 4× smaller |
| `model_fp16.onnx` | ~67 MB | FP16, for GPU/NPU |

The INT8 model is the recommended production artifact — accuracy loss is negligible for this task.

---

## Synthetic Dataset

No real labeled receipts were used. The training data is fully synthetic.

**Generation pipeline** (`notebooks/receipt_ocr_qa_pipeline.ipynb`, Section 1):

1. Ground truth values are generated (station name, address, date, fuel type, prices, etc.)
2. A receipt is rendered using one of 5 structured templates
3. Realistic OCR noise is applied (char confusion, digit transposition, space merge/split, punct substitution)
4. Answer spans are recorded by position tracking during construction — zero alignment failures
5. Output: SQuAD v2 format JSON

**5 receipt templates:**

| Template | Layout style | Example use case |
|----------|-------------|-----------------|
| `vertical` | Labeled rows, one field per line | Most Indian/Asian receipts |
| `compact` | Pipe-separated header, single item line | Thermal printer receipts |
| `table` | Column-aligned with dashes | Gas station column-format |
| `minimal` | Values only, no labels | Simple POS receipts |
| `verbose` | Fully labeled, bordered | Formal tax invoices |

**OCR noise model** — 6 noise types weighted by frequency in real OCR output:

| Noise type | Weight | Example |
|-----------|--------|---------|
| Char confusion | 30% | `0` ↔ `O`, `1` ↔ `l`, `5` ↔ `S` |
| Digit adjacent shift | 20% | `102.45` → `102.46` |
| Space insert | 13% | `Unleaded` → `Unle aded` |
| Space delete | 13% | `25.500 Ltr` → `25.500Ltr` |
| Punct substitution | 12% | `.` ↔ `,`, `:` ↔ `;` |
| Case flip | 12% | `Shell` → `sHell` |
| Digit transposition | separate pass | `25.500` → `52.500` |

**Dataset stats (default config):**
- 10,000 synthetic receipts generated
- ~80,000 QA pairs across 8 fields
- 72k train / 8k validation split (90/10)
- 100% valid spans (position tracking guarantees no corrupted labels)

---

## Fields extracted

| Field key | Question asked | Example answer |
|-----------|---------------|----------------|
| `station_name` | What is the name of the fuel station? | `Shell India Pump` |
| `station_addr` | What is the address of the fuel station? | `251 High Street, Mumbai, MH` |
| `date` | What is the date of the transaction? | `15/08/2023 10:30` |
| `currency` | What is the currency symbol or code? | `₹` |
| `fuel_type` | What type of fuel was purchased? | `Regular Unleaded` |
| `unit_price` | What is the price per unit of fuel? | `102.45` |
| `volume` | What volume of fuel was purchased? | `25.500` |
| `total_amount` | What is the total amount paid? | `2612.47` |

---

## Notebook walkthrough

All stages live in `notebooks/receipt_ocr_qa_pipeline.ipynb`.

| Section | Cells | What it does |
|---------|-------|-------------|
| **1. Data generation** | 03–06 | Generates synthetic SQuAD dataset |
| **2. Fine-tuning** | 08–11 | Loads data, tokenizes, trains model |
| **3. ONNX export** | 13 | Exports to FP32 ONNX → INT8 → FP16 |
| **4a. Dev inference** | 16 | Transformers pipeline on a receipt image |
| **4b. Prod inference** | 17 | ONNX runtime + custom tokenizer, no transformers dep |

---

## Limitations

This is a prototype trained entirely on synthetic data. Expect degraded performance on:

- **Non-fuel receipts** — grocery, restaurant, hotel receipts use completely different field names and layouts
- **Low-quality images** — blurry, rotated, crumpled receipts degrade OCR upstream; the QA model can't recover from bad OCR
- **Mixed-script receipts** — Indian receipts often mix Hindi/regional script with English; OCR handles this poorly
- **Unseen layouts** — receipts with extra sections (GST breakdown, loyalty points, vehicle/pump number) confuse the model
- **Very short values** — single-char currency symbols (₹, £, €) have lower extraction confidence

**To improve for production:** fine-tune on 200–500 real labeled receipts on top of the synthetic baseline. The [SROIE](https://rrc.cvc.uab.es/?ch=13) and [CORD](https://github.com/clovaai/cord) datasets are public labeled receipt benchmarks worth including.

---

## Requirements

```
torch
transformers
datasets
evaluate
pytesseract
Pillow
safetensors
onnx
onnxruntime
onnxruntime-tools
onnxconverter-common
```

System dependency (Tesseract binary):
```bash
apt-get install -y tesseract-ocr    # Linux
brew install tesseract              # macOS
```

Install Python deps:
```bash
pip install torch transformers datasets evaluate pytesseract Pillow \
    safetensors onnx onnxruntime onnxruntime-tools onnxconverter-common
```

Production inference only (no training deps):
```bash
pip install pytesseract Pillow onnxruntime
```
