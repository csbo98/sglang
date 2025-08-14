#!/usr/bin/env python3
"""Simple test script for modality limits functionality."""

import sys
import os
sys.path.insert(0, '/workspace/python')

# Test the core modality limits module without full dependencies
def test_modality_limits_core():
    """Test the core modality limit functionality."""
    
    # Import just the modality limits module
    from sglang.srt.multimodal.mm_utils import flatten_nested_list, has_valid_data
    
    print("Testing flatten_nested_list...")
    # Test flatten_nested_list
    assert flatten_nested_list([1, 2, 3]) == [1, 2, 3]
    assert flatten_nested_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert flatten_nested_list([[[1]], [[2, 3]]]) == [1, 2, 3]
    print("✓ flatten_nested_list works correctly")
    
    print("\nTesting has_valid_data...")
    # Test has_valid_data
    assert has_valid_data(None) == False
    assert has_valid_data([]) == False
    assert has_valid_data([None]) == False
    assert has_valid_data("data") == True
    assert has_valid_data(["data"]) == True
    assert has_valid_data([["data"]]) == True
    print("✓ has_valid_data works correctly")
    
    # Now test the modality limits module
    from sglang.srt.multimodal.modality_limits import (
        ModalityLimitConfig,
        ModalityLimitValidator,
    )
    
    print("\nTesting ModalityLimitConfig...")
    # Test config creation
    config = ModalityLimitConfig(image_limit=5, video_limit=2, audio_limit=3)
    assert config.image_limit == 5
    assert config.video_limit == 2
    assert config.audio_limit == 3
    assert config.has_limits() == True
    print("✓ ModalityLimitConfig creation works")
    
    # Test from_dict
    config2 = ModalityLimitConfig.from_dict({"image": 10, "video": 1, "audio": 5})
    assert config2.image_limit == 10
    assert config2.video_limit == 1
    assert config2.audio_limit == 5
    print("✓ ModalityLimitConfig.from_dict works")
    
    print("\nTesting ModalityLimitValidator...")
    # Test validator
    validator = ModalityLimitValidator(config)
    
    # Test within limits - should not raise
    try:
        validator.validate_request(
            image_data=["img1.jpg", "img2.jpg"],
            video_data="video.mp4",
            audio_data=["audio1.wav", "audio2.wav"]
        )
        print("✓ Validation passes for inputs within limits")
    except ValueError as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    # Test exceeding limits - should raise
    try:
        validator.validate_request(
            image_data=["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]
        )
        print("✗ Should have raised ValueError for exceeding image limit")
        return False
    except ValueError as e:
        if "exceeds the maximum limit" in str(e):
            print("✓ Validation correctly rejects inputs exceeding limits")
        else:
            print(f"✗ Unexpected error message: {e}")
            return False
    
    # Test summary
    summary = validator.get_summary()
    assert "images: 5" in summary
    assert "videos: 2" in summary
    assert "audio: 3" in summary
    print("✓ get_summary works correctly")
    
    print("\n" + "="*50)
    print("All core tests passed successfully!")
    print("="*50)
    return True

if __name__ == "__main__":
    success = test_modality_limits_core()
    sys.exit(0 if success else 1)