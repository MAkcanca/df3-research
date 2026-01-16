#!/usr/bin/env python3
"""
CFA (Color Filter Array) Analyzer Tool Tests

Tests the CFA consistency analysis tool based on the Popescu-Farid CFA detection algorithm.
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
from src.tools.forensic import perform_cfa_detection
from src.tools.forensic import cfa_tools as cfa_tools_module


@pytest.fixture
def test_image():
    """Create a test image with varied texture for CFA analysis."""
    np.random.seed(42)
    width, height = 512, 512
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
    test_image.save(tmp_path, 'JPEG', quality=90)
    yield tmp_path
    os.unlink(tmp_path)


class TestCFABasicFunctionality:
    """Test basic CFA functionality and output structure."""

    def test_returns_expected_json_structure(self, jpeg_file):
        """Test that perform_cfa_detection returns expected JSON structure."""
        result_str = perform_cfa_detection(jpeg_file)
        result = json.loads(result_str)

        required_fields = [
            'tool', 'status', 'image_path', 'analysis_channel', 'window_size',
            'window_count', 'pattern', 'has_cfa_signal', 'interpretation',
            'cfa_consistency_score', 'window_populations', 'distribution',
            'm_value_stats', 'strongest_cfa_windows', 'weakest_cfa_windows'
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_status_completed(self, jpeg_file):
        """Test that analysis completes successfully."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        assert result['status'] == 'completed'

    def test_tool_name(self, jpeg_file):
        """Test that tool name is correct."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        assert result['tool'] == 'perform_cfa_detection'


class TestCFAStatistics:
    """Test CFA statistics validity."""

    def test_m_value_statistics(self, jpeg_file):
        """Test that M value statistics are valid."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        stats = result['m_value_stats']

        required_stats = ['min', 'max', 'mean', 'median', 'std']
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
            assert isinstance(stats[stat], (int, float))
            assert np.isfinite(stats[stat])

    def test_distribution_analysis(self, jpeg_file):
        """Test that distribution analysis is valid."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        dist = result['distribution']

        assert 'type' in dist
        assert 'is_bimodal' in dist
        assert isinstance(dist['is_bimodal'], bool)

    def test_window_populations(self, jpeg_file):
        """Test that window populations are valid."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        pop = result['window_populations']

        required_pop = ['textured_with_cfa', 'flat_no_texture', 'intermediate', 'textured_pct', 'flat_pct']
        for field in required_pop:
            assert field in pop, f"Missing population field: {field}"

    def test_cfa_consistency_score_range(self, jpeg_file):
        """Test that CFA consistency score is in valid range."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        score = result.get('cfa_consistency_score')

        if score is not None:
            assert isinstance(score, (int, float))
            assert np.isfinite(score)
            assert 0.0 <= score <= 1.0


class TestCFAParameters:
    """Test CFA parameter handling."""

    def test_window_size_parameter(self, jpeg_file):
        """Test that window size parameter affects results."""
        window_sizes = [128, 256, 512]
        results = []

        for window in window_sizes:
            result = json.loads(perform_cfa_detection(
                json.dumps({"image_path": jpeg_file, "window": window})
            ))
            if result['status'] == 'completed':
                results.append((window, result['window_count']))

        assert len(results) == len(window_sizes)
        # Smaller windows should produce more windows
        assert results[0][1] >= results[-1][1]

    def test_pattern_parameter(self, jpeg_file):
        """Test that pattern parameter works for all patterns."""
        patterns = ['RGGB', 'GRBG', 'GBRG', 'BGGR']

        for pattern in patterns:
            result = json.loads(perform_cfa_detection(
                json.dumps({"image_path": jpeg_file, "pattern": pattern})
            ))
            assert result['status'] == 'completed'
            assert result['pattern'] == pattern

    def test_channel_parameter(self, jpeg_file):
        """Test that channel parameter works for R, G, B."""
        channels = ['R', 'G', 'B']

        for channel in channels:
            result = json.loads(perform_cfa_detection(
                json.dumps({"image_path": jpeg_file, "channel": channel})
            ))
            assert result['status'] == 'completed'
            assert result['analysis_channel'].upper() == channel.upper()

    def test_top_k_parameter(self, jpeg_file):
        """Test that top_k parameter controls window counts."""
        top_k_values = [3, 5, 10]

        for top_k in top_k_values:
            result = json.loads(perform_cfa_detection(
                json.dumps({"image_path": jpeg_file, "top_k": top_k})
            ))
            assert result['status'] == 'completed'
            assert len(result['strongest_cfa_windows']) <= top_k
            assert len(result['weakest_cfa_windows']) <= top_k


class TestCFAInputParsing:
    """Test CFA input parsing."""

    def test_plain_path_input(self, jpeg_file):
        """Test plain path input."""
        result = json.loads(perform_cfa_detection(jpeg_file))
        assert result['status'] == 'completed'

    def test_json_input(self, jpeg_file):
        """Test JSON input with parameters."""
        result = json.loads(perform_cfa_detection(json.dumps({
            "image_path": jpeg_file,
            "window": 256,
            "pattern": "RGGB",
            "channel": "G"
        })))
        assert result['status'] == 'completed'
        assert result['window_size'] == 256
        assert result['pattern'] == 'RGGB'


class TestCFAErrorHandling:
    """Test CFA error handling."""

    def test_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = json.loads(perform_cfa_detection("/nonexistent/path/image.jpg"))
        assert 'error' in result


class TestCFAImageFormats:
    """Test CFA with different image formats."""

    def test_jpeg_format(self, test_image):
        """Test JPEG format."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'JPEG')
        result = json.loads(perform_cfa_detection(tmp_path))
        os.unlink(tmp_path)
        assert result['status'] == 'completed'

    def test_png_format(self, test_image):
        """Test PNG format."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_path = f.name
        test_image.save(tmp_path, 'PNG')
        result = json.loads(perform_cfa_detection(tmp_path))
        os.unlink(tmp_path)
        assert result['status'] == 'completed'


class TestCFAImageSizes:
    """Test CFA with different image sizes."""

    @pytest.mark.parametrize("size", [(256, 256), (512, 512), (1024, 512), (512, 1024)])
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
        pil_img.save(tmp_path, 'JPEG', quality=90)
        result = json.loads(perform_cfa_detection(tmp_path))
        os.unlink(tmp_path)
        assert result['status'] == 'completed'


class TestCFABimodalityMath:
    """Test bimodality coefficient math."""

    def test_normal_distribution(self):
        """Test BC for normal distribution (should be ~0.33)."""
        rng = np.random.default_rng(0)
        normal = rng.standard_normal(10000) + 10.0
        res = cfa_tools_module._detect_bimodality(normal.tolist())
        bc = float(res.get("bimodality_coefficient", float("nan")))

        assert np.isfinite(bc)
        assert not res.get("is_bimodal", True)
        assert 0.28 <= bc <= 0.38

    def test_bimodal_distribution(self):
        """Test BC for clearly bimodal distribution (should be >0.6)."""
        two_point = np.array([-1.0, 1.0] * 5000, dtype=np.float64)
        res = cfa_tools_module._detect_bimodality(two_point.tolist())
        bc = float(res.get("bimodality_coefficient", float("nan")))

        assert np.isfinite(bc)
        assert res.get("is_bimodal", False)
        assert bc > 0.6

    def test_uniform_distribution(self):
        """Test BC for uniform distribution (should be ~0.555)."""
        uniform = np.linspace(0.0, 1.0, 10000, dtype=np.float64)
        res = cfa_tools_module._detect_bimodality(uniform.tolist())
        bc = float(res.get("bimodality_coefficient", float("nan")))

        assert np.isfinite(bc)
        assert not res.get("is_bimodal", True)
        assert 0.50 <= bc <= 0.60
