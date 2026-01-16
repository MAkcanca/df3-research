#!/usr/bin/env python3
"""
ELA (Error Level Analysis) Tool Tests

Tests the ELA forensic tool based on the Sherloq ELA algorithm.
"""

import base64
import json
import os
import sys
import tempfile
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.forensic import perform_ela


@pytest.fixture
def test_image():
    """Create a test image with varied texture for ELA analysis."""
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


class TestELABasicFunctionality:
    """Test basic ELA functionality and output structure."""

    def test_returns_expected_json_structure(self, jpeg_file):
        """Test that perform_ela returns expected JSON structure."""
        result_str = perform_ela(jpeg_file)
        result = json.loads(result_str)

        required_fields = ['tool', 'status', 'image_path', 'quality',
                           'ela_mean', 'ela_std', 'ela_anomaly_score']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_status_completed(self, jpeg_file):
        """Test that analysis completes successfully."""
        result = json.loads(perform_ela(jpeg_file))
        assert result['status'] == 'completed'

    def test_tool_name(self, jpeg_file):
        """Test that tool name is correct."""
        result = json.loads(perform_ela(jpeg_file))
        assert result['tool'] == 'perform_ela'


class TestELAStatistics:
    """Test ELA statistics validity."""

    def test_ela_mean_valid(self, jpeg_file):
        """Test that ELA mean is valid."""
        result = json.loads(perform_ela(jpeg_file))
        ela_mean = result['ela_mean']

        assert isinstance(ela_mean, (int, float))
        assert np.isfinite(ela_mean)
        assert ela_mean >= 0

    def test_ela_std_valid(self, jpeg_file):
        """Test that ELA std is valid."""
        result = json.loads(perform_ela(jpeg_file))
        ela_std = result['ela_std']

        assert isinstance(ela_std, (int, float))
        assert np.isfinite(ela_std)
        assert ela_std >= 0

    def test_ela_anomaly_score_valid(self, jpeg_file):
        """Test that ELA anomaly score is valid."""
        result = json.loads(perform_ela(jpeg_file))
        ela_anomaly_score = result['ela_anomaly_score']

        assert isinstance(ela_anomaly_score, (int, float))
        assert np.isfinite(ela_anomaly_score)


class TestELAMapGeneration:
    """Test ELA map generation."""

    def test_map_generated_when_requested(self, jpeg_file):
        """Test that ELA map is generated when requested."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "return_map": True})
        ))

        assert 'ela_map' in result
        assert result['ela_map'] is not None

    def test_map_is_valid_png(self, jpeg_file):
        """Test that ELA map is a valid PNG."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "return_map": True})
        ))

        ela_map = result['ela_map']
        assert ela_map.startswith('data:image/png;base64,')

        b64_data = ela_map.split(',', 1)[1]
        png_data = base64.b64decode(b64_data)
        assert png_data.startswith(b'\x89PNG\r\n\x1a\n')

        # Verify it can be loaded as an image
        map_img = Image.open(BytesIO(png_data))
        assert map_img.size[0] > 0 and map_img.size[1] > 0

    def test_map_not_generated_when_not_requested(self, jpeg_file):
        """Test that ELA map is not generated when not requested."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "return_map": False})
        ))

        assert result.get('ela_map') is None


class TestELAParameters:
    """Test ELA parameter handling."""

    def test_quality_parameter_applied(self, jpeg_file):
        """Test that quality parameter is applied."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "quality": 85, "return_map": False})
        ))

        assert result['quality'] == 85

    def test_quality_clamping_low(self, jpeg_file):
        """Test that low quality values are clamped."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "quality": -10, "return_map": False})
        ))

        assert result['quality'] == 1

    def test_quality_clamping_high(self, jpeg_file):
        """Test that high quality values are clamped."""
        result = json.loads(perform_ela(
            json.dumps({"path": jpeg_file, "quality": 150, "return_map": False})
        ))

        assert result['quality'] == 100


class TestELAInputParsing:
    """Test ELA input parsing."""

    def test_plain_path_input(self, jpeg_file):
        """Test plain path input."""
        result = json.loads(perform_ela(jpeg_file))
        assert result['status'] == 'completed'

    def test_json_input(self, jpeg_file):
        """Test JSON input with parameters."""
        result = json.loads(perform_ela(json.dumps({
            "path": jpeg_file,
            "quality": 85,
            "max_size": 256,
            "return_map": False
        })))
        assert result['status'] == 'completed'
        assert result['quality'] == 85


class TestELAErrorHandling:
    """Test ELA error handling."""

    def test_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = json.loads(perform_ela("/nonexistent/path/image.jpg"))
        assert result['status'] == 'error'
        assert 'error' in result


class TestELAImageFormats:
    """Test ELA with different image formats."""

    def test_jpeg_format(self, test_image):
        """Test JPEG format."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG')
        result = json.loads(perform_ela(tmp_path))
        os.unlink(tmp_path)
        assert result['status'] == 'completed'

    def test_png_format(self, test_image):
        """Test PNG format."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'PNG')
        result = json.loads(perform_ela(tmp_path))
        os.unlink(tmp_path)
        assert result['status'] == 'completed'


class TestELAMapResizing:
    """Test ELA map resizing."""

    def test_map_resized_when_needed(self):
        """Test that map is resized when image exceeds max_size."""
        np.random.seed(42)
        width, height = 512, 512
        x = np.linspace(0, 255, width)
        y = np.linspace(0, 255, height)
        xx, yy = np.meshgrid(x, y)
        img = ((xx + yy) / 2).astype(np.uint8)
        pil_img = Image.fromarray(img, mode='L').convert('RGB')

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        pil_img.save(tmp_path, 'JPEG', quality=75)

        result = json.loads(perform_ela(
            json.dumps({"path": tmp_path, "max_size": 256, "return_map": True})
        ))
        os.unlink(tmp_path)

        if result.get('ela_map_size'):
            max_dim = max(result['ela_map_size'])
            assert max_dim <= 256
