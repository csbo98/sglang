#!/usr/bin/env python3
"""
Convenience script to run the enhanced multi-modal CI tests.
This script addresses the requirements from issue #8496.

Usage:
    python3 run_multimodal_ci_tests.py --test-type all
    python3 run_multimodal_ci_tests.py --test-type tp2
    python3 run_multimodal_ci_tests.py --test-type tp4
    python3 run_multimodal_ci_tests.py --test-type audio
    python3 run_multimodal_ci_tests.py --test-type stress
"""

import argparse
import subprocess
import sys
import time
from typing import List, Dict


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, cwd=".", capture_output=False)
        end_time = time.time()
        print(f"\n✅ SUCCESS: {description} (took {end_time - start_time:.2f}s)")
        return True
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"\n❌ FAILED: {description} (took {end_time - start_time:.2f}s)")
        print(f"Error code: {e.returncode}")
        return False


def get_test_commands() -> Dict[str, List[Dict[str, any]]]:
    """Get all test commands organized by category."""
    return {
        "tp2": [
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_single_image_chat_completion_tp2"],
                "desc": "TP=2 Basic Single Image Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_stress_test_with_memory_monitoring"],
                "desc": "TP=2 Stress Test with Memory Monitoring"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp2.TestQwen2_5_VLServerTP2.test_high_throughput_image_processing_tp2"],
                "desc": "TP=2 High Throughput Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp2.TestMinicpmoServerTP2.test_mixed_audio_visual_tp2"],
                "desc": "TP=2 Mixed Audio/Visual Test"
            }
        ],
        "tp4": [
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_single_image_chat_completion_tp4"],
                "desc": "TP=4 Basic Single Image Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_large_batch_stress_test"],
                "desc": "TP=4 Large Batch Stress Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_throughput_benchmark_tp4"],
                "desc": "TP=4 Throughput Benchmark"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp4.TestQwen2_5_VLServerTP4.test_sustained_load_test_tp4"],
                "desc": "TP=4 Sustained Load Test"
            }
        ],
        "audio": [
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_audio_transcription_stress"],
                "desc": "Audio Transcription Stress Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_mixed_audio_visual_batch"],
                "desc": "Mixed Audio/Visual Batch Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_audio_memory_leak_detection"],
                "desc": "Audio Memory Leak Detection"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingTP2.test_audio_processing_tp2_performance"],
                "desc": "Audio TP=2 Performance Test"
            }
        ],
        "stress": [
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_stress_test_with_memory_monitoring"],
                "desc": "TP=2 Memory Stress Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_large_batch_stress_test"],
                "desc": "TP=4 Large Batch Stress Test"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_audio_memory_leak_detection"],
                "desc": "Audio Memory Leak Detection"
            },
            {
                "cmd": ["python3", "-m", "unittest", "test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_concurrent_audio_processing_scalability"],
                "desc": "Audio Concurrent Processing Scalability"
            }
        ],
        "suite": [
            {
                "cmd": ["python3", "run_suite.py", "--suite", "per-commit-2-gpu", "--range-begin", "7", "--range-end", "8"],
                "desc": "Run TP=2 Test Suite (per-commit-2-gpu subset)"
            },
            {
                "cmd": ["python3", "run_suite.py", "--suite", "per-commit-4-gpu", "--range-begin", "3", "--range-end", "4"],
                "desc": "Run TP=4 Test Suite (per-commit-4-gpu subset)"
            },
            {
                "cmd": ["python3", "run_suite.py", "--suite", "per-commit", "--range-begin", "108", "--range-end", "109"],
                "desc": "Run Audio Test Suite (per-commit subset)"
            }
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run enhanced multi-modal CI tests for issue #8496",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  tp2     - Run TP=2 (Tensor Parallelism) tests
  tp4     - Run TP=4 (Tensor Parallelism) tests  
  audio   - Run expanded audio processing tests
  stress  - Run stress tests with memory leak detection
  suite   - Run test suites (integration with CI)
  all     - Run all enhanced tests

Examples:
  python3 run_multimodal_ci_tests.py --test-type tp2
  python3 run_multimodal_ci_tests.py --test-type stress --continue-on-failure
  python3 run_multimodal_ci_tests.py --test-type all --dry-run
        """
    )
    
    parser.add_argument(
        "--test-type", 
        choices=["tp2", "tp4", "audio", "stress", "suite", "all"],
        required=True,
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running tests even if some fail"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="Print commands without executing them"
    )
    
    args = parser.parse_args()
    
    test_commands = get_test_commands()
    
    # Determine which test categories to run
    if args.test_type == "all":
        categories_to_run = ["tp2", "tp4", "audio", "stress"]
    else:
        categories_to_run = [args.test_type]
    
    print(f"\n🚀 Enhanced Multi-Modal CI Test Runner")
    print(f"📋 Test Categories: {', '.join(categories_to_run)}")
    print(f"⚙️  Continue on Failure: {args.continue_on_failure}")
    print(f"🔍 Dry Run: {args.dry_run}")
    
    total_tests = sum(len(test_commands[cat]) for cat in categories_to_run)
    print(f"📊 Total Tests: {total_tests}")
    
    if args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - Commands that would be executed:")
        print(f"{'='*60}")
        
        for category in categories_to_run:
            print(f"\n--- {category.upper()} Tests ---")
            for test in test_commands[category]:
                print(f"  {test['desc']}")
                print(f"    Command: {' '.join(test['cmd'])}")
        
        return 0
    
    # Run the tests
    passed = 0
    failed = 0
    failed_tests = []
    
    start_time = time.time()
    
    for category in categories_to_run:
        print(f"\n🔧 Running {category.upper()} Tests...")
        
        for test in test_commands[category]:
            success = run_command(test["cmd"], test["desc"])
            
            if success:
                passed += 1
            else:
                failed += 1
                failed_tests.append(test["desc"])
                
                if not args.continue_on_failure:
                    print(f"\n❌ Stopping due to failure in: {test['desc']}")
                    break
        
        if failed > 0 and not args.continue_on_failure:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time:.2f} seconds")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%" if (passed+failed) > 0 else "N/A")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"  - {test}")
    
    if failed == 0:
        print(f"\n🎉 All tests passed! Multi-modal CI enhancements are working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())