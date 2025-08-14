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
"""Modality limit validation for multimodal inputs."""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from sglang.srt.utils import flatten_nested_list
from sglang.srt.multimodal.mm_utils import has_valid_data


@dataclass
class ModalityLimitConfig:
    """Configuration for per-modality input limits."""
    
    image_limit: Optional[int] = None
    video_limit: Optional[int] = None
    audio_limit: Optional[int] = None
    
    @classmethod
    def from_dict(cls, limits_dict: Dict[str, Optional[int]]) -> "ModalityLimitConfig":
        """Create a ModalityLimitConfig from a dictionary."""
        return cls(
            image_limit=limits_dict.get("image"),
            video_limit=limits_dict.get("video"),
            audio_limit=limits_dict.get("audio"),
        )
    
    def has_limits(self) -> bool:
        """Check if any modality limits are configured."""
        return any([
            self.image_limit is not None,
            self.video_limit is not None,
            self.audio_limit is not None,
        ])


class ModalityLimitValidator:
    """Validator for enforcing per-modality input limits."""
    
    def __init__(self, config: Optional[ModalityLimitConfig] = None):
        """
        Initialize the validator with a configuration.
        
        Args:
            config: ModalityLimitConfig instance or None if no limits
        """
        self.config = config or ModalityLimitConfig()
    
    def validate_request(
        self,
        image_data: Optional[Union[List, object]] = None,
        video_data: Optional[Union[List, object]] = None,
        audio_data: Optional[Union[List, object]] = None,
    ) -> None:
        """
        Validate that the multimodal inputs don't exceed configured limits.
        
        Args:
            image_data: Image input data (can be single item, list, or nested list)
            video_data: Video input data (can be single item, list, or nested list)
            audio_data: Audio input data (can be single item, list, or nested list)
            
        Raises:
            ValueError: If any modality exceeds its configured limit
        """
        if not self.config.has_limits():
            return
        
        # Validate image inputs
        if self.config.image_limit is not None and has_valid_data(image_data):
            image_count = self._count_items(image_data)
            if image_count > self.config.image_limit:
                raise ValueError(
                    f"Number of images ({image_count}) exceeds the maximum limit "
                    f"of {self.config.image_limit} images per prompt. "
                    f"Please reduce the number of images or adjust the limit using "
                    f"--max-images-per-prompt or --limit-mm-per-prompt."
                )
        
        # Validate video inputs
        if self.config.video_limit is not None and has_valid_data(video_data):
            video_count = self._count_items(video_data)
            if video_count > self.config.video_limit:
                raise ValueError(
                    f"Number of videos ({video_count}) exceeds the maximum limit "
                    f"of {self.config.video_limit} videos per prompt. "
                    f"Please reduce the number of videos or adjust the limit using "
                    f"--max-videos-per-prompt or --limit-mm-per-prompt."
                )
        
        # Validate audio inputs
        if self.config.audio_limit is not None and has_valid_data(audio_data):
            audio_count = self._count_items(audio_data)
            if audio_count > self.config.audio_limit:
                raise ValueError(
                    f"Number of audio inputs ({audio_count}) exceeds the maximum limit "
                    f"of {self.config.audio_limit} audio inputs per prompt. "
                    f"Please reduce the number of audio inputs or adjust the limit using "
                    f"--max-audios-per-prompt or --limit-mm-per-prompt."
                )
    
    def _count_items(self, data: Optional[Union[List, object]]) -> int:
        """
        Count the number of items in the data.
        
        Handles single items, lists, and nested lists.
        """
        if data is None:
            return 0
        
        if not isinstance(data, list):
            # Single item
            return 1
        
        # Flatten nested lists and count items
        flattened = flatten_nested_list(data)
        return len([item for item in flattened if item is not None])
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the configured limits."""
        if not self.config.has_limits():
            return "No modality limits configured"
        
        limits = []
        if self.config.image_limit is not None:
            limits.append(f"images: {self.config.image_limit}")
        if self.config.video_limit is not None:
            limits.append(f"videos: {self.config.video_limit}")
        if self.config.audio_limit is not None:
            limits.append(f"audio: {self.config.audio_limit}")
        
        return f"Modality limits: {', '.join(limits)}"