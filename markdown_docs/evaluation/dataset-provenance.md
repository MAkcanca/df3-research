# Dataset Provenance

Documentation of the evaluation dataset, including sources and sampling methodology.

---

## Evaluation Dataset

DF3 evaluation results are only scientifically comparable when the **dataset fingerprint (digest) matches**. This repo currently contains **multiple benchmark datasets**, including:

- A mixed **synthetic/real** dataset (GenImage/DRAGON/Nano-banana/etc.)
- A **FaceForensics++ (FF++)** frame dataset derived from real deepfake videos (deepfake-oriented domain)

| Property | Value |
|----------|-------|
| **Dataset Fingerprint** | `sha256(sorted(ids))[:16]` |
| **Compare runs when** | Digests match |

### Sample Limit Configurations

| Dataset | Limit | Digest | Location | Notes |
|-------|--------|-------|
| Synthetic mix | n=500 | `f987165daff0de70` | `data2/samples.jsonl` | 500 samples (247 fake, 253 real) |
| Synthetic mix | n=200 | `1f78e35118013ed4` | `data2/samples.jsonl` | Subset of the same pool (first 200 by stable order) |
| FaceForensics++ frames | n=500 | `c02071eee1ee544a` | `data/ffpp_500.jsonl` | 500 samples (**444 fake, 56 real**); PNG frames extracted from FF++ videos |

The n=200 runs use the first 200 samples from the same shuffled dataset. Results from different limits should be compared with caution due to sample composition differences.

---

## Source Datasets

### Synthetic Images (Fake Class)

**GenImage**

- Source: [GenImage GitHub](https://github.com/GenImage-Dataset/GenImage)
- Reference: "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image"
- Structure: Organized by generator (Midjourney, Stable Diffusion, etc.)
- Note: The "nature" subset contains real images from ImageNet

**DRAGON**

- Source: [Hugging Face](https://huggingface.co/datasets/lesc-unifi/dragon)

**Nano-banana 150k**

- Sources:
  - [Nano-consistent-150k](https://huggingface.co/datasets/Yejy53/Nano-consistent-150k)
  - [Nano-banana-150k](https://huggingface.co/datasets/bitmind/Nano-banana-150k)

### Real Images (Real Class)

**GenImage Nature Subset**

- Real images collected from ImageNet
- Paired with synthetic images in the GenImage dataset

**Nano-banana Real Subset**

- Real images as provided by dataset hosting

---

## Sampling Methodology

### FaceForensics++ (FF++) frames dataset

This dataset is built from FaceForensics++ videos by extracting frames (PNG, no re-encode) and sampling frames into a JSONL manifest.

**Frame extraction**

```powershell
python scripts/extract_frames_from_videos.py --input_dir FaceForensicsPP --output_dir FaceForensicsPP_frames --frame_positions 0.0 0.5
```

**Dataset creation**

```powershell
python scripts/create_faceforensicspp_dataset.py --frames_dir FaceForensicsPP_frames --output data/ffpp_500.jsonl --limit 500 --seed 42
```

!!! important "Class imbalance and interpretation"
    The current FF++ sample is **highly imbalanced** (444 fake / 56 real). Prefer reporting **balanced accuracy**, **MCC**, and class-specific rates (e.g., `real_false_flag_rate`) alongside overall accuracy.

### Sampling Command

```powershell
python scripts/sample_dataset.py \
    --num_samples 500 \
    --output_dir samples \
    --dataset_dir .\dataset\ \
    --seed 42
```

### Folder Discovery

| Class | Folder Names |
|-------|-------------|
| Real | `real`, `nature`, `authentic`, `genuine`, `real_images` |
| Fake | `fake`, `ai`, `synthetic`, `generated`, `deepfake`, `fake_images` |

### Process

1. Recursive enumeration of images (jpg, jpeg, png, bmp, tiff)
2. Random sampling with fixed seed for reproducibility
3. Copy to output directory with canonical names
4. Generate JSONL manifest
5. Shuffle entries

### Output Format

```json
{"id": "sample-001", "image": "sample-001.jpg", "label": "real"}
```

Both `image`/`label` and `image_path`/`ground_truth` field variants are supported.

### Preventing Label Leakage

!!! important "No Label Information in Paths"
    File paths and filenames must use neutral identifiers (e.g., `sample-001.jpg`, `image_123.jpg`) and must not contain label-related terms like "fake", "real", "deepfake", "synthetic", "authentic", etc. This prevents LLMs from cheating by reading file metadata instead of analyzing image content.

---

## Dataset Digest Computation

```python
import hashlib
sample_ids = sorted([r["id"] for r in dataset])
digest = hashlib.sha256(str(sample_ids).encode()).hexdigest()[:16]
```

Compare results only when digests match.

---

## Licensing

| Dataset | License |
|---------|---------|
| GenImage | See repository |
| DRAGON | See Hugging Face card |
| Nano-banana | See Hugging Face card |
| ImageNet | Restricted research use |

For external delivery, consider manifest-with-hashes approach for restricted sources.

---

## See Also

- [Methodology](methodology.md) — Evaluation framework
- [Benchmark Results](results.md) — Performance data
