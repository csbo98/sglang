#!/usr/bin/env python3
"""
Example script demonstrating per-modality input limits in SGLang.

This example shows how to configure and use modality limits to control
the maximum number of images, videos, and audio inputs per prompt.
"""

import argparse
import asyncio
import json
from typing import List

import sglang as sgl


def create_multimodal_request_example():
    """Example of creating multimodal requests with different input counts."""
    
    # Example 1: Single image request (always allowed)
    single_image_request = {
        "text": "What do you see in this image?",
        "image_data": "path/to/image.jpg",
    }
    
    # Example 2: Multiple images request
    multi_image_request = {
        "text": "Compare these images and describe the differences.",
        "image_data": [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            "path/to/image3.jpg",
        ],
    }
    
    # Example 3: Mixed modality request
    mixed_modality_request = {
        "text": "Analyze this multimedia content.",
        "image_data": ["image1.jpg", "image2.jpg"],
        "video_data": "video.mp4",
        "audio_data": ["audio1.wav", "audio2.wav"],
    }
    
    return [single_image_request, multi_image_request, mixed_modality_request]


def print_server_launch_examples():
    """Print examples of launching the server with different modality limits."""
    
    print("=" * 80)
    print("EXAMPLES: Launching SGLang Server with Modality Limits")
    print("=" * 80)
    print()
    
    print("1. Using individual modality limits:")
    print("   python -m sglang.launch_server \\")
    print("       --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("       --enable-multimodal \\")
    print("       --max-images-per-prompt 5 \\")
    print("       --max-videos-per-prompt 1 \\")
    print("       --max-audios-per-prompt 3")
    print()
    
    print("2. Using JSON configuration for limits:")
    print("   python -m sglang.launch_server \\")
    print("       --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("       --enable-multimodal \\")
    print('       --limit-mm-per-prompt \'{"image": 10, "video": 2, "audio": 5}\'')
    print()
    
    print("3. Combining JSON and individual limits (individual overrides JSON):")
    print("   python -m sglang.launch_server \\")
    print("       --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("       --enable-multimodal \\")
    print('       --limit-mm-per-prompt \'{"image": 10, "video": 2, "audio": 5}\' \\')
    print("       --max-images-per-prompt 20  # This overrides the JSON value")
    print()
    
    print("4. Strict single-image mode (common for production):")
    print("   python -m sglang.launch_server \\")
    print("       --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \\")
    print("       --enable-multimodal \\")
    print("       --max-images-per-prompt 1 \\")
    print("       --max-videos-per-prompt 0 \\")
    print("       --max-audios-per-prompt 0")
    print()


async def test_modality_limits():
    """Test function to demonstrate modality limit validation."""
    
    print("=" * 80)
    print("TESTING: Modality Limit Validation")
    print("=" * 80)
    print()
    
    # Simulated test cases
    test_cases = [
        {
            "name": "Within limits",
            "images": 3,
            "videos": 1,
            "audios": 2,
            "limits": {"image": 5, "video": 2, "audio": 3},
            "should_pass": True,
        },
        {
            "name": "Exceeds image limit",
            "images": 10,
            "videos": 1,
            "audios": 1,
            "limits": {"image": 5, "video": 2, "audio": 3},
            "should_pass": False,
        },
        {
            "name": "Exceeds video limit",
            "images": 2,
            "videos": 3,
            "audios": 1,
            "limits": {"image": 5, "video": 2, "audio": 3},
            "should_pass": False,
        },
        {
            "name": "No limits configured",
            "images": 100,
            "videos": 50,
            "audios": 75,
            "limits": {},
            "should_pass": True,
        },
    ]
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"  Input: {test['images']} images, {test['videos']} videos, {test['audios']} audios")
        print(f"  Limits: {test['limits']}")
        print(f"  Expected: {'✓ Pass' if test['should_pass'] else '✗ Fail (should be rejected)'}")
        print()


def demonstrate_error_messages():
    """Show example error messages when limits are exceeded."""
    
    print("=" * 80)
    print("EXAMPLE ERROR MESSAGES")
    print("=" * 80)
    print()
    
    print("When image limit is exceeded:")
    print('  ValueError: Number of images (10) exceeds the maximum limit of 5 images per prompt.')
    print('  Please reduce the number of images or adjust the limit using --max-images-per-prompt')
    print('  or --limit-mm-per-prompt.')
    print()
    
    print("When video limit is exceeded:")
    print('  ValueError: Number of videos (3) exceeds the maximum limit of 1 videos per prompt.')
    print('  Please reduce the number of videos or adjust the limit using --max-videos-per-prompt')
    print('  or --limit-mm-per-prompt.')
    print()
    
    print("When audio limit is exceeded:")
    print('  ValueError: Number of audio inputs (5) exceeds the maximum limit of 3 audio inputs per prompt.')
    print('  Please reduce the number of audio inputs or adjust the limit using --max-audios-per-prompt')
    print('  or --limit-mm-per-prompt.')
    print()


def show_use_cases():
    """Display common use cases for modality limits."""
    
    print("=" * 80)
    print("COMMON USE CASES")
    print("=" * 80)
    print()
    
    use_cases = [
        {
            "scenario": "Production API with memory constraints",
            "config": "--max-images-per-prompt 1 --max-videos-per-prompt 0",
            "reason": "Prevents OOM by limiting to single image processing",
        },
        {
            "scenario": "Document analysis service",
            "config": "--max-images-per-prompt 10 --max-videos-per-prompt 0 --max-audios-per-prompt 0",
            "reason": "Allows multiple document pages but blocks video/audio",
        },
        {
            "scenario": "Video summarization service",
            "config": "--max-images-per-prompt 0 --max-videos-per-prompt 1 --max-audios-per-prompt 0",
            "reason": "Processes one video at a time to manage GPU memory",
        },
        {
            "scenario": "Multimodal chatbot with balanced limits",
            "config": '--limit-mm-per-prompt \'{"image": 5, "video": 1, "audio": 3}\'',
            "reason": "Provides flexibility while preventing resource exhaustion",
        },
        {
            "scenario": "Development/testing environment",
            "config": "(no limits specified)",
            "reason": "Allows unlimited inputs for experimentation",
        },
    ]
    
    for i, case in enumerate(use_cases, 1):
        print(f"{i}. {case['scenario']}")
        print(f"   Configuration: {case['config']}")
        print(f"   Reason: {case['reason']}")
        print()


def main():
    """Main function to run all demonstrations."""
    
    parser = argparse.ArgumentParser(
        description="Demonstrate SGLang per-modality input limits"
    )
    parser.add_argument(
        "--show-launch-examples",
        action="store_true",
        help="Show server launch examples",
    )
    parser.add_argument(
        "--show-error-messages",
        action="store_true",
        help="Show example error messages",
    )
    parser.add_argument(
        "--show-use-cases",
        action="store_true",
        help="Show common use cases",
    )
    parser.add_argument(
        "--test-validation",
        action="store_true",
        help="Run validation tests",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all demonstrations",
    )
    
    args = parser.parse_args()
    
    # Default to showing all if no specific option is selected
    if not any([
        args.show_launch_examples,
        args.show_error_messages,
        args.show_use_cases,
        args.test_validation,
    ]):
        args.show_all = True
    
    print("\n" + "=" * 80)
    print("SGLang Per-Modality Input Limits - Feature Demonstration")
    print("=" * 80 + "\n")
    
    if args.show_launch_examples or args.show_all:
        print_server_launch_examples()
        print()
    
    if args.show_use_cases or args.show_all:
        show_use_cases()
        print()
    
    if args.test_validation or args.show_all:
        asyncio.run(test_modality_limits())
        print()
    
    if args.show_error_messages or args.show_all:
        demonstrate_error_messages()
        print()
    
    print("=" * 80)
    print("For more information, see the SGLang documentation.")
    print("=" * 80)


if __name__ == "__main__":
    main()