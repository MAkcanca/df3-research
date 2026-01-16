# Dataset Provenance

Documentation of the evaluation dataset, including sources and sampling methodology.

---

## Evaluation Dataset

The evaluation uses a single dataset sampled from multiple sources. Different runs use different sample limits (n=200 or n=500) but draw from the same underlying sample pool.

| Property | Value |
|----------|-------|
| **Total Samples** | 500 (247 fake, 253 real) |
| **ID Digest** | Computed from sorted sample IDs |
| **Location** | `data2/samples.jsonl` |
| **Format Mix** | ~80% JPEG, ~20% PNG |

### Sample Limit Configurations

| Limit | Digest | Usage |
|-------|--------|-------|
| n=500 | `f987165daff0de70` | Full dataset runs |
| n=200 | `1f78e35118013ed4` | Subset (first 200 by stable order) |

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
