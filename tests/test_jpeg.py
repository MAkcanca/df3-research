#!/usr/bin/env python3
"""
JPEG Forensic Tools Tests

Tests JPEG forensic tools based on MATLAB reference implementations.
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
from src.tools.forensic import detect_jpeg_quantization


@pytest.fixture
def test_image():
    """Create a test image with varied texture for DCT analysis."""
    np.random.seed(42)
    width, height = 256, 256
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


class TestJPEGQualityEstimation:
    """Test JPEG quality estimation."""

    @pytest.mark.parametrize("quality", [50, 70, 85, 95])
    def test_quality_estimation_accuracy(self, test_image, quality):
        """Test that quality estimation is accurate within 2 levels."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG', quality=quality)

        result = json.loads(detect_jpeg_quantization(tmp_path))
        os.unlink(tmp_path)

        est_q = result['quality_estimates']['0']['estimated_quality']
        error = abs(est_q - quality)
        assert error <= 2, f"Quality estimation error {error} > 2 for Q={quality}"


class TestSacScore:
    """Test Sac score (JPEGness measurement)."""

    def test_sac_score_valid(self, jpeg_file):
        """Test that Sac score is valid."""
        result = json.loads(detect_jpeg_quantization(jpeg_file))
        sac_score = result['sac_score']['score']

        assert np.isfinite(sac_score)

    @pytest.mark.parametrize("quality", [50, 70, 90])
    def test_sac_scores_for_different_qualities(self, test_image, quality):
        """Test Sac scores for different JPEG qualities."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG', quality=quality)

        result = json.loads(detect_jpeg_quantization(tmp_path))
        os.unlink(tmp_path)

        sac_score = result['sac_score']['score']
        assert np.isfinite(sac_score)


class TestBlockMap:
    """Test block-level analysis."""

    def test_block_map_generated(self, jpeg_file):
        """Test that block map is generated when requested."""
        result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": jpeg_file, "include": {"block_map": True, "per_frequency": False}})
        ))

        assert 'block_map' in result
        block_map = np.array(result['block_map']['map'])
        assert block_map.size > 0

    def test_block_map_uniform_for_single_compression(self, jpeg_file):
        """Test that block map is relatively uniform for uniformly compressed image."""
        result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": jpeg_file, "include": {"block_map": True, "per_frequency": False}})
        ))

        block_map = np.array(result['block_map']['map'])
        std_prob = block_map.std()
        # For uniformly compressed image, expect relatively low variance
        assert std_prob < 0.5


class TestDoubleCompression:
    """Test double compression detection."""

    def test_double_compression_analysis(self, test_image):
        """Test analysis of double-compressed image."""
        # Single compression at Q90
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            single_path = f.name
        test_image.save(single_path, 'JPEG', quality=90)

        # Double compression: Q70 -> Q90
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
            tmp_q70 = f1.name
        test_image.save(tmp_q70, 'JPEG', quality=70)
        img_q70 = Image.open(tmp_q70)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f2:
            double_path = f2.name
        img_q70.save(double_path, 'JPEG', quality=90)
        img_q70.close()
        os.unlink(tmp_q70)

        # Analyze both
        single_result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": single_path, "include": {"block_map": True, "per_frequency": False}})
        ))
        double_result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": double_path, "include": {"block_map": True, "per_frequency": False}})
        ))

        os.unlink(single_path)
        os.unlink(double_path)

        # Both should complete successfully
        assert 'sac_score' in single_result
        assert 'sac_score' in double_result


class TestPrimaryTableEstimation:
    """Test primary quantization table estimation."""

    def test_primary_table_generated(self, jpeg_file):
        """Test that primary table estimation is generated."""
        result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": jpeg_file, "include": {"primary_table": True}})
        ))

        assert 'estimated_primary_quantization' in result

    @pytest.mark.parametrize("quality", [50, 70, 85, 95])
    def test_primary_table_accuracy(self, test_image, quality):
        """Test primary table estimation accuracy."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG', quality=quality)

        result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": tmp_path, "include": {"primary_table": True}})
        ))
        os.unlink(tmp_path)

        if 'estimated_primary_quantization' in result:
            est = np.array(result['estimated_primary_quantization']['table'])
            act = np.array(result['quantization_tables']['0'])

            # DC coefficient should match exactly
            assert est[0, 0] == act[0, 0]


class TestTruncationMask:
    """Test truncation mask for odd-dimension images."""

    def test_truncation_mask_shape(self):
        """Test truncation mask shape for non-8-multiple dimensions."""
        np.random.seed(42)
        width, height = 250, 310
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        xx, yy = np.meshgrid(x, y)
        img = ((xx + yy) / 2).astype(np.uint8)
        pil_img = Image.fromarray(img, mode='L').convert('RGB')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        pil_img.save(tmp_path, 'JPEG', quality=80)

        result = json.loads(detect_jpeg_quantization(
            json.dumps({"path": tmp_path, "include": {"primary_table": True, "truncation_mask": True}})
        ))
        os.unlink(tmp_path)

        if 'estimated_primary_quantization' in result:
            mask = np.array(result['estimated_primary_quantization'].get('mask', []))
            if mask.size > 0:
                h_blocks = height // 8
                w_blocks = width // 8
                assert mask.shape == (h_blocks, w_blocks)


class TestJpeglibStatus:
    """Test jpeglib availability reporting."""

    def test_jpeglib_status_reported(self, jpeg_file):
        """Test that jpeglib availability is reported."""
        result = json.loads(detect_jpeg_quantization(jpeg_file))

        assert 'jpeglib_available' in result
        assert 'coefficient_source' in result
