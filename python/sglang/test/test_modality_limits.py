# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for modality limit validation functionality."""

import json
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.multimodal.modality_limits import (
    ModalityLimitConfig,
    ModalityLimitValidator,
)
from sglang.srt.server_args import ServerArgs


class TestModalityLimitConfig(unittest.TestCase):
    """Test cases for ModalityLimitConfig."""

    def test_create_config_with_all_limits(self):
        """Test creating config with all modality limits."""
        config = ModalityLimitConfig(
            image_limit=5,
            video_limit=2,
            audio_limit=3,
        )
        self.assertEqual(config.image_limit, 5)
        self.assertEqual(config.video_limit, 2)
        self.assertEqual(config.audio_limit, 3)
        self.assertTrue(config.has_limits())

    def test_create_config_with_partial_limits(self):
        """Test creating config with only some modality limits."""
        config = ModalityLimitConfig(image_limit=10)
        self.assertEqual(config.image_limit, 10)
        self.assertIsNone(config.video_limit)
        self.assertIsNone(config.audio_limit)
        self.assertTrue(config.has_limits())

    def test_create_config_with_no_limits(self):
        """Test creating config with no limits."""
        config = ModalityLimitConfig()
        self.assertIsNone(config.image_limit)
        self.assertIsNone(config.video_limit)
        self.assertIsNone(config.audio_limit)
        self.assertFalse(config.has_limits())

    def test_from_dict(self):
        """Test creating config from dictionary."""
        limits_dict = {
            "image": 5,
            "video": 2,
            "audio": 3,
        }
        config = ModalityLimitConfig.from_dict(limits_dict)
        self.assertEqual(config.image_limit, 5)
        self.assertEqual(config.video_limit, 2)
        self.assertEqual(config.audio_limit, 3)

    def test_from_dict_with_extra_keys(self):
        """Test that extra keys in dict are ignored."""
        limits_dict = {
            "image": 5,
            "text": 100,  # Should be ignored
            "unknown": 50,  # Should be ignored
        }
        config = ModalityLimitConfig.from_dict(limits_dict)
        self.assertEqual(config.image_limit, 5)
        self.assertIsNone(config.video_limit)
        self.assertIsNone(config.audio_limit)


class TestModalityLimitValidator(unittest.TestCase):
    """Test cases for ModalityLimitValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModalityLimitConfig(
            image_limit=3,
            video_limit=1,
            audio_limit=2,
        )
        self.validator = ModalityLimitValidator(self.config)

    def test_validate_within_limits(self):
        """Test validation passes when within limits."""
        # Single items
        self.validator.validate_request(
            image_data="image1.jpg",
            video_data="video1.mp4",
            audio_data="audio1.wav",
        )

        # Lists within limits
        self.validator.validate_request(
            image_data=["img1.jpg", "img2.jpg"],
            video_data=["video1.mp4"],
            audio_data=["audio1.wav", "audio2.wav"],
        )

    def test_validate_exceeds_image_limit(self):
        """Test validation fails when image limit is exceeded."""
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_request(
                image_data=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"],
            )
        self.assertIn("Number of images (4) exceeds", str(cm.exception))
        self.assertIn("maximum limit of 3", str(cm.exception))

    def test_validate_exceeds_video_limit(self):
        """Test validation fails when video limit is exceeded."""
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_request(
                video_data=["video1.mp4", "video2.mp4"],
            )
        self.assertIn("Number of videos (2) exceeds", str(cm.exception))
        self.assertIn("maximum limit of 1", str(cm.exception))

    def test_validate_exceeds_audio_limit(self):
        """Test validation fails when audio limit is exceeded."""
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_request(
                audio_data=["audio1.wav", "audio2.wav", "audio3.wav"],
            )
        self.assertIn("Number of audio inputs (3) exceeds", str(cm.exception))
        self.assertIn("maximum limit of 2", str(cm.exception))

    def test_validate_nested_lists(self):
        """Test validation with nested list inputs."""
        # Nested lists should be flattened for counting
        with self.assertRaises(ValueError) as cm:
            self.validator.validate_request(
                image_data=[["img1.jpg", "img2.jpg"], ["img3.jpg", "img4.jpg"]],
            )
        self.assertIn("Number of images (4) exceeds", str(cm.exception))

    def test_validate_with_none_data(self):
        """Test validation with None data."""
        self.validator.validate_request(
            image_data=None,
            video_data=None,
            audio_data=None,
        )

    def test_validate_with_empty_lists(self):
        """Test validation with empty lists."""
        self.validator.validate_request(
            image_data=[],
            video_data=[],
            audio_data=[],
        )

    def test_validate_no_limits_configured(self):
        """Test that validation passes when no limits are configured."""
        validator = ModalityLimitValidator(ModalityLimitConfig())
        # Should not raise any errors
        validator.validate_request(
            image_data=["img1.jpg"] * 100,
            video_data=["video1.mp4"] * 50,
            audio_data=["audio1.wav"] * 75,
        )

    def test_get_summary(self):
        """Test getting human-readable summary."""
        summary = self.validator.get_summary()
        self.assertIn("images: 3", summary)
        self.assertIn("videos: 1", summary)
        self.assertIn("audio: 2", summary)

        # Test with no limits
        validator = ModalityLimitValidator(ModalityLimitConfig())
        summary = validator.get_summary()
        self.assertEqual(summary, "No modality limits configured")


class TestServerArgsModalityLimits(unittest.TestCase):
    """Test cases for ServerArgs modality limit parsing."""

    def test_parse_json_limits(self):
        """Test parsing JSON modality limits."""
        args = ServerArgs(
            model_path="test-model",
            limit_mm_per_prompt='{"image": 5, "video": 2, "audio": 3}',
        )
        limits = args.parse_modality_limits()
        self.assertEqual(limits["image"], 5)
        self.assertEqual(limits["video"], 2)
        self.assertEqual(limits["audio"], 3)

    def test_parse_individual_limits(self):
        """Test individual modality limit arguments."""
        args = ServerArgs(
            model_path="test-model",
            max_images_per_prompt=10,
            max_videos_per_prompt=5,
            max_audios_per_prompt=8,
        )
        limits = args.parse_modality_limits()
        self.assertEqual(limits["image"], 10)
        self.assertEqual(limits["video"], 5)
        self.assertEqual(limits["audio"], 8)

    def test_individual_limits_override_json(self):
        """Test that individual limits override JSON configuration."""
        args = ServerArgs(
            model_path="test-model",
            limit_mm_per_prompt='{"image": 5, "video": 2, "audio": 3}',
            max_images_per_prompt=20,  # This should override the JSON value
        )
        limits = args.parse_modality_limits()
        self.assertEqual(limits["image"], 20)  # Overridden
        self.assertEqual(limits["video"], 2)   # From JSON
        self.assertEqual(limits["audio"], 3)   # From JSON

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        args = ServerArgs(
            model_path="test-model",
            limit_mm_per_prompt='invalid json',
        )
        # Should not raise, but return empty dict
        limits = args.parse_modality_limits()
        self.assertEqual(limits, {})

    def test_parse_case_insensitive_keys(self):
        """Test that JSON keys are normalized to lowercase."""
        args = ServerArgs(
            model_path="test-model",
            limit_mm_per_prompt='{"Image": 5, "VIDEO": 2, "Audio": 3}',
        )
        limits = args.parse_modality_limits()
        self.assertEqual(limits["image"], 5)
        self.assertEqual(limits["video"], 2)
        self.assertEqual(limits["audio"], 3)

    def test_negative_limit_validation(self):
        """Test that negative limits raise an error."""
        args = ServerArgs(
            model_path="test-model",
            max_images_per_prompt=-1,
        )
        with self.assertRaises(ValueError) as cm:
            args.parse_modality_limits()
        self.assertIn("must be non-negative", str(cm.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests for modality limits with TokenizerManager."""

    @patch('sglang.srt.managers.tokenizer_manager.ModelConfig')
    @patch('sglang.srt.managers.tokenizer_manager.get_tokenizer')
    def test_tokenizer_manager_with_limits(self, mock_get_tokenizer, mock_model_config):
        """Test that TokenizerManager properly initializes with modality limits."""
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.server_args import PortArgs

        # Mock the model config
        mock_config_instance = MagicMock()
        mock_config_instance.is_multimodal = False
        mock_config_instance.is_generation = True
        mock_config_instance.is_image_gen = False
        mock_config_instance.context_len = 2048
        mock_config_instance.image_token_id = None
        mock_model_config.from_server_args.return_value = mock_config_instance

        # Create server args with modality limits
        server_args = ServerArgs(
            model_path="test-model",
            max_images_per_prompt=5,
            max_videos_per_prompt=2,
            max_audios_per_prompt=3,
        )
        port_args = MagicMock(spec=PortArgs)

        # Initialize TokenizerManager
        manager = TokenizerManager(server_args, port_args)

        # Check that modality validator is initialized
        self.assertIsNotNone(manager.modality_validator)
        self.assertEqual(manager.modality_validator.config.image_limit, 5)
        self.assertEqual(manager.modality_validator.config.video_limit, 2)
        self.assertEqual(manager.modality_validator.config.audio_limit, 3)


if __name__ == "__main__":
    unittest.main()