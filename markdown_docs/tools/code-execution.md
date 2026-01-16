# Code Execution Tool

Dynamic Python code execution for custom forensic analysis.

---

## Overview

The `execute_python_code` tool enables the agent to write and execute arbitrary Python code for specialized analysis not covered by the standard tools.

---

## Purpose

### Use Cases

- Custom statistical analysis
- Image manipulation operations
- Specialized forensic techniques
- Exploratory data analysis
- Verification of hypotheses

### When Used

The agent may invoke code execution when:

- Standard tools don't address a specific question
- Custom computation is needed
- Complex analysis requires programming
- Results need verification

---

## Capabilities

### Available Libraries

The code execution environment includes:

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `PIL/Pillow` | Image processing |
| `scipy` | Scientific computing |
| `cv2` (OpenCV) | Computer vision |
| `json` | Data serialization |
| `math` | Mathematical functions |

### Input/Output

**Input:**

- Python code as string
- Image path available to code

**Output:**

```json
{
    "tool": "execute_python_code",
    "status": "completed",
    "stdout": "Analysis result: 0.85",
    "stderr": "",
    "return_value": {"score": 0.85}
}
```

---

## Example Usage

### Custom Pixel Analysis

```python
# Agent-generated code example
from PIL import Image
import numpy as np

img = Image.open(image_path)
arr = np.array(img)

# Check for unusual pixel value distributions
hist, _ = np.histogram(arr.flatten(), bins=256)
entropy = -np.sum(hist[hist>0] / hist.sum() * np.log2(hist[hist>0] / hist.sum()))

print(f"Pixel entropy: {entropy:.2f}")
```

### Edge Detection Analysis

```python
import cv2
import numpy as np

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, 100, 200)

edge_density = edges.sum() / edges.size
print(f"Edge density: {edge_density:.4f}")
```

### Color Distribution Check

```python
from PIL import Image
import numpy as np

img = Image.open(image_path).convert('RGB')
arr = np.array(img)

# Check for unusual color clustering
r_std = arr[:,:,0].std()
g_std = arr[:,:,1].std()
b_std = arr[:,:,2].std()

print(f"R std: {r_std:.2f}, G std: {g_std:.2f}, B std: {b_std:.2f}")
```

---

## Security Considerations

### Sandboxing

!!! warning "Execution Environment"
    Code executes with access to the local filesystem and system resources. In production deployments, consider containerization or sandboxing.

### Current Protections

- Timeout limits prevent infinite loops
- Output size limits prevent memory exhaustion
- Standard library restrictions (configurable)

### Risks

| Risk | Mitigation |
|------|------------|
| File system access | Restrict to data directory |
| Network access | Consider blocking |
| Resource exhaustion | Timeout and memory limits |
| Code injection | Agent generates code, not user |

---

## Forensic Applications

### Verifying Tool Results

```python
# Manual verification of ELA-like analysis
from PIL import Image
import numpy as np

original = Image.open(image_path)
original.save('/tmp/recompressed.jpg', 'JPEG', quality=75)
recompressed = Image.open('/tmp/recompressed.jpg')

diff = np.abs(np.array(original).astype(float) - np.array(recompressed).astype(float))
print(f"Mean difference: {diff.mean():.2f}")
print(f"Max difference: {diff.max():.2f}")
```

### Custom Feature Extraction

```python
import cv2
import numpy as np

img = cv2.imread(image_path)

# Detect and count faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print(f"Detected {len(faces)} faces")
```

### Statistical Tests

```python
from scipy import stats
import numpy as np
from PIL import Image

img = np.array(Image.open(image_path).convert('L'))

# Test for uniform noise (Gaussian test)
_, p_value = stats.normaltest(img.flatten())
print(f"Normality test p-value: {p_value:.4f}")
```

---

## Agent Behavior

### When Agent Uses Code Execution

1. **Gap in tools** — No standard tool addresses the question
2. **Verification** — Double-checking other tool results
3. **Exploration** — Investigating specific hypothesis
4. **Custom metrics** — Computing domain-specific measures

### Agent Code Quality

The agent generates code that:

- Uses available libraries
- Handles errors gracefully (usually)
- Outputs interpretable results
- Avoids infinite loops (usually)

### Limitations

Agent-generated code may:

- Have bugs or errors
- Not be optimal
- Miss edge cases
- Produce hard-to-interpret output

---

## Output Interpretation

### Successful Execution

```json
{
    "status": "completed",
    "stdout": "Result: 0.85",
    "stderr": "",
    "execution_time": 0.23
}
```

### Failed Execution

```json
{
    "status": "error",
    "stdout": "",
    "stderr": "NameError: name 'undefined_var' is not defined",
    "execution_time": 0.01
}
```

### Timeout

```json
{
    "status": "timeout",
    "stdout": "Partial output...",
    "stderr": "Execution exceeded time limit",
    "execution_time": 30.0
}
```

---

## Best Practices

### For Analysis

- Treat code execution results as supplementary
- Verify unusual findings manually
- Consider execution errors as inconclusive

### For Security

- Review execution logs periodically
- Consider sandboxing in production
- Limit available libraries if needed
- Monitor resource usage

### For Reproducibility

- Log all executed code
- Save intermediate results
- Document any custom analysis

---

## Configuration

### Timeout

Default: 30 seconds

### Memory Limit

Inherits from Python process

### Available Libraries

Configured in tool implementation

---

## See Also

- [Tools Overview](overview.md) — All forensic tools
- [Agent Pipeline](../architecture/agent-pipeline.md) — How agent uses tools
- [Troubleshooting](../reference/troubleshooting.md) — Error handling
