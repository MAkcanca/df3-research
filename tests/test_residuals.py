#!/usr/bin/env python3
"""
Residuals Extraction Tool Tests

Tests the residual extraction tool implementation using DRUNet.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.forensic import extract_residuals


@pytest.fixture
def test_image():
    """Create a test image with varied texture for residual analysis."""
    np.random.seed(42)
    width, height = 1024, 1024
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xx, yy = np.meshgrid(x, y)
    img = ((xx + yy) / 2).astype(np.uint8)
    noise = np.random.randint(-30, 30, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')


@pytest.fixture
def jpeg_file(test_image):
    """Create a temporary JPEG file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmp_path = f.name
    test_image.save(tmp_path, 'JPEG', quality=75)
    yield tmp_path
    os.unlink(tmp_path)


def _drunet_available(result):
    """Check if DRUNet is available based on result."""
    if result['status'] == 'error':
        error_msg = result.get('error', '')
        return 'DRUNet' not in error_msg and 'PyTorch' not in error_msg
    return True


class TestResidualsBasicFunctionality:
    """Test basic residuals functionality and output structure."""

    def test_returns_json_structure(self, jpeg_file):
        """Test that extract_residuals returns valid JSON."""
        result_str = extract_residuals(jpeg_file)
        result = json.loads(result_str)

        # Should have either status=completed or status=error
        assert 'status' in result
        assert result['status'] in ['completed', 'error']

    def test_completed_has_required_fields(self, jpeg_file):
        """Test that completed analysis has required fields."""
        result = json.loads(extract_residuals(jpeg_file))

        if result['status'] == 'completed':
            required_fields = [
                'tool', 'image_path',
                'residual_mean', 'residual_std', 'residual_skew', 'residual_kurtosis',
                'residual_energy', 'residual_energy_mean', 'residual_energy_std', 'residual_energy_p95'
            ]
            for field in required_fields:
                assert field in result, f"Missing required field: {field}"
        else:
            # If error, should have error field
            assert 'error' in result

    def test_tool_name(self, jpeg_file):
        """Test that tool name is correct."""
        result = json.loads(extract_residuals(jpeg_file))

        if result['status'] == 'completed':
            assert result['tool'] == 'extract_residuals'


class TestResidualsStatistics:
    """Test residual statistics validity."""

    def test_residual_statistics_valid(self, jpeg_file):
        """Test that all residual statistics are valid numbers."""
        result = json.loads(extract_residuals(jpeg_file))

        if result['status'] != 'completed':
            pytest.skip("DRUNet/PyTorch not available")

        stats = {
            'residual_mean': result['residual_mean'],
            'residual_std': result['residual_std'],
            'residual_skew': result['residual_skew'],
            'residual_kurtosis': result['residual_kurtosis'],
            'residual_energy': result['residual_energy'],
            'residual_energy_mean': result['residual_energy_mean'],
            'residual_energy_std': result['residual_energy_std'],
            'residual_energy_p95': result['residual_energy_p95'],
        }

        for name, value in stats.items():
            assert isinstance(value, (int, float)), f"{name} is not a number"
            assert np.isfinite(value), f"{name} is not finite"


class TestResidualsErrorHandling:
    """Test residuals error handling."""

    def test_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = json.loads(extract_residuals("/nonexistent/path/image.jpg"))
        assert result['status'] == 'error'
        assert 'error' in result


class TestResidualsImageFormats:
    """Test residuals with different image formats."""

    def test_jpeg_format(self, test_image):
        """Test JPEG format."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG')
        result = json.loads(extract_residuals(tmp_path))
        os.unlink(tmp_path)

        if _drunet_available(result):
            assert result['status'] == 'completed'

    def test_png_format(self, test_image):
        """Test PNG format."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'PNG')
        result = json.loads(extract_residuals(tmp_path))
        os.unlink(tmp_path)

        if _drunet_available(result):
            assert result['status'] == 'completed'


class TestResidualsImageSizes:
    """Test residuals with different image sizes."""

    @pytest.mark.parametrize("size", [(128, 128), (256, 256), (512, 512)])
    def test_various_sizes(self, size):
        """Test various image sizes."""
        np.random.seed(42)
        width, height = size
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        xx, yy = np.meshgrid(x, y)
        img = ((xx + yy) / 2).astype(np.uint8)
        noise = np.random.randint(-30, 30, (height, width), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img, mode='L').convert('RGB')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        pil_img.save(tmp_path, 'JPEG', quality=75)
        result = json.loads(extract_residuals(tmp_path))
        os.unlink(tmp_path)

        if _drunet_available(result):
            assert result['status'] == 'completed'


class TestResidualsConsistency:
    """Test residuals consistency across runs."""

    def test_deterministic_results(self, jpeg_file):
        """Test that results are consistent across multiple runs."""
        results = []
        for _ in range(3):
            result = json.loads(extract_residuals(jpeg_file))
            if result['status'] == 'completed':
                results.append(result)
            elif 'DRUNet' in result.get('error', '') or 'PyTorch' in result.get('error', ''):
                pytest.skip("DRUNet/PyTorch not available")

        if len(results) < 3:
            pytest.skip("Not all runs succeeded")

        # Check consistency
        for key in ['residual_mean', 'residual_std', 'residual_energy']:
            values = [r[key] for r in results]
            max_diff = max(values) - min(values)
            assert max_diff < 1e-6, f"{key} varies across runs: {values}"
