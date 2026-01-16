# Delivery Checklist

Checklist for preparing DF3 deliverables for external labs or publication.

---

## Pre-Delivery Checklist

### Code & Environment

- [ ] **Repository clean** — No debug code, commented blocks, or sensitive data
- [ ] **Dependencies documented** — `requirements.txt` with pinned versions
- [ ] **Installation tested** — Fresh environment setup verified
- [ ] **README updated** — Current instructions, no stale info
- [ ] **License included** — Appropriate license file

### Documentation

- [ ] **User guide complete** — All workflows documented
- [ ] **API reference current** — Matches actual code
- [ ] **Examples working** — All code examples tested
- [ ] **Limitations documented** — Honest capability assessment

### Data & Datasets

- [ ] **Dataset provenance** — Sources and licenses documented
- [ ] **Sample data included** — Test images for verification
- [ ] **Manifests provided** — If full data can't be shared
- [ ] **Privacy checked** — No PII or sensitive content

### Evaluation Results

- [ ] **Results reproducible** — Re-run and verify
- [ ] **Methodology documented** — All parameters recorded
- [ ] **Metrics explained** — Definitions and interpretations
- [ ] **Confidence intervals** — Uncertainty quantified

---

## Code Review Checklist

### Security

- [ ] No hardcoded credentials
- [ ] API keys loaded from environment
- [ ] No sensitive data in logs
- [ ] Input validation present

### Quality

- [ ] No TODO comments unaddressed
- [ ] No commented-out code blocks
- [ ] Consistent code style
- [ ] Meaningful variable names

### Functionality

- [ ] All scripts executable
- [ ] Error handling present
- [ ] Edge cases considered
- [ ] Resource cleanup (files, connections)

---

## Documentation Checklist

### Required Documents

| Document | Purpose | Status |
|----------|---------|--------|
| README.md | Quick start | [ ] |
| INSTALL.md | Detailed setup | [ ] |
| LICENSE | Legal terms | [ ] |
| CHANGELOG.md | Version history | [ ] |
| CONTRIBUTING.md | Contribution guide | [ ] |

### User Documentation

| Section | Content | Status |
|---------|---------|--------|
| Quick Start | 5-minute setup | [ ] |
| Installation | Full environment setup | [ ] |
| User Guide | Workflow tutorials | [ ] |
| API Reference | Function documentation | [ ] |
| Troubleshooting | Common issues | [ ] |
| FAQ | Frequent questions | [ ] |

### Technical Documentation

| Section | Content | Status |
|---------|---------|--------|
| Architecture | System design | [ ] |
| Data Flow | Processing pipeline | [ ] |
| Tool Reference | Forensic tool details | [ ] |
| Evaluation | Methodology & metrics | [ ] |
| Reproducibility | Reproduction guide | [ ] |
| Limitations | Honest assessment | [ ] |

---

## Dataset Delivery Checklist

### If Including Full Dataset

- [ ] License permits redistribution
- [ ] Sources credited properly
- [ ] No copyright violations
- [ ] Privacy-safe content

### If Manifest-Only

- [ ] Retrieval instructions provided
- [ ] SHA256 hashes for verification
- [ ] Source URLs documented
- [ ] Sampling methodology described

### Dataset Documentation

- [ ] Sample counts verified
- [ ] Class balance documented
- [ ] Source datasets listed
- [ ] Processing steps described

---

## Evaluation Delivery Checklist

### Results Artifacts

- [ ] Raw results file (`.jsonl`)
- [ ] Aggregated metrics (`.json`)
- [ ] Summary tables (`.md`)
- [ ] Configuration used (`.json`)

### Metadata

- [ ] Model versions recorded
- [ ] Dataset digest included
- [ ] Timestamp documented
- [ ] Git commit recorded

### Validation

- [ ] Metrics recomputable from raw results
- [ ] Sample spot-checked manually
- [ ] Edge cases reviewed
- [ ] Error cases analyzed

---

## Final Quality Checks

### Fresh Install Test

```powershell
# Clone fresh copy
git clone $REPO_URL test-install
cd test-install

# Create environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install
pip install -r requirements.txt

# Run basic test
python scripts/analyze_image.py --image data/sample.jpg

# Run evaluation sample
python scripts/evaluate_llms.py --dataset data/sample.jsonl --limit 10
```

### Documentation Accuracy

- [ ] All commands execute successfully
- [ ] All file paths exist
- [ ] All links resolve
- [ ] Screenshots current

### Completeness

- [ ] All listed features work
- [ ] All documented APIs functional
- [ ] All examples produce expected output
- [ ] All configuration options work

---

## Packaging Options

### Option A: GitHub Repository

```
Deliverable: Repository URL + access
Contents:
  - Full source code
  - Documentation in markdown_docs/
  - Sample data
  - Results artifacts
```

### Option B: Zip Archive

```
df3-v1.0.0.zip
├── src/
├── scripts/
├── markdown_docs/
├── data/sample/
├── results/
├── requirements.txt
├── README.md
├── LICENSE
└── INSTALL.md
```

### Option C: Docker Container

```dockerfile
# Dockerfile for complete environment
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "scripts/analyze_image.py", "--help"]
```

---

## Handoff Documentation

### Cover Letter Contents

1. **Overview** — What DF3 is and does
2. **Key Capabilities** — Main features
3. **Setup Instructions** — Quick start reference
4. **Known Limitations** — Honest assessment
5. **Support Contact** — How to get help

### Training Materials (if needed)

- [ ] Video walkthrough
- [ ] Live demo session
- [ ] Q&A documentation
- [ ] Office hours schedule

---

## Post-Delivery

### Support Plan

- [ ] Issue tracking set up
- [ ] Response time defined
- [ ] Escalation path documented
- [ ] Maintenance schedule planned

### Version Control

- [ ] Release tagged
- [ ] Changelog updated
- [ ] Version number bumped
- [ ] Dependencies frozen

---

## Quick Pre-Delivery Commands

```powershell
# Check for secrets
Select-String -Pattern "sk-|api[_-]key|password" -Path src/**/*.py

# Check for TODOs
Select-String -Pattern "TODO|FIXME|XXX" -Path src/**/*.py

# Verify requirements
pip check

# Run tests
python -m pytest tests/

# Build docs
mkdocs build

# Check for large files
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 10MB}
```

---

## See Also

- [Installation](../getting-started/installation.md) — Setup guide
- [Reproducibility](reproducibility.md) — Reproduction requirements
- [Limitations](limitations.md) — Known issues
