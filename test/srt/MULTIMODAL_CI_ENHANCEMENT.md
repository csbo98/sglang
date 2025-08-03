# Multi-Modal CI Enhancement

This document describes the enhancements made to the multi-modal CI testing suite to address [issue #8496](https://github.com/sgl-project/sglang/issues/8496).

## Overview

The following improvements have been implemented:

1. **Tensor Parallelism (TP) Support**: Added TP-2 and TP-4 test configurations
2. **Stress Testing**: Implemented comprehensive stress tests with memory leak detection
3. **Audio Expansion**: Extended testing beyond vision to include comprehensive audio processing
4. **CI Integration**: Updated test suites to include new test configurations

## New Test Files

### 1. `test_vision_openai_server_tp2.py`
**Purpose**: Test multi-modal models with TP=2 configuration
**Key Features**:
- Basic functionality tests (single image, multi-image, video) with TP=2
- Stress testing with memory monitoring
- Concurrent multimodal request handling
- High throughput image processing tests
- Mixed audio/visual processing (for audio-capable models)

**Test Classes**:
- `TestQwen2VLServerTP2`: Qwen2-VL with TP=2
- `TestQwen2_5_VLServerTP2`: Qwen2.5-VL with TP=2  
- `TestMinicpmoServerTP2`: MiniCPM-o with TP=2 (includes audio)

**Memory Monitoring**: 
- Tracks system memory usage during stress tests
- Detects potential memory leaks with configurable thresholds
- Provides detailed memory statistics

### 2. `test_vision_openai_server_tp4.py`
**Purpose**: Test multi-modal models with TP=4 configuration
**Key Features**:
- Large batch stress testing optimized for TP=4
- Advanced memory monitoring with GPU memory tracking
- High concurrency mixed modality tests
- Throughput benchmarking
- Sustained load testing over extended periods

**Test Classes**:
- `TestQwen2VLServerTP4`: Qwen2-VL with TP=4
- `TestQwen2_5_VLServerTP4`: Qwen2.5-VL with TP=4

**Advanced Features**:
- GPU memory monitoring alongside system memory
- Higher concurrency levels (16+ concurrent requests)
- Throughput assertions and performance benchmarks

### 3. `test_audio_multimodal_expanded.py`
**Purpose**: Comprehensive audio testing beyond current vision-only focus
**Key Features**:
- Audio transcription stress testing
- Mixed audio/visual batch processing
- Audio quality analysis
- Audio format robustness testing
- Audio-specific memory leak detection
- Concurrent audio processing scalability tests
- TP=2 audio performance testing

**Test Classes**:
- `TestAudioProcessingExpanded`: Comprehensive audio testing
- `TestAudioProcessingTP2`: Audio processing with TP=2

**Audio-Specific Monitoring**:
- Specialized `AudioMemoryMonitor` class
- Event logging for audio processing correlation
- More frequent memory sampling for audio workloads

## Memory Monitoring Features

### MemoryMonitor Class
- Basic memory usage tracking
- Peak memory detection
- Memory leak threshold assertions
- Background monitoring with thread safety

### AdvancedMemoryMonitor Class
- System and GPU memory tracking
- Comprehensive statistics reporting
- Higher sampling frequency
- Extended monitoring capabilities

### AudioMemoryMonitor Class
- Audio processing event correlation
- Specialized for audio workload patterns
- Event logging for debugging

## CI Integration

### Test Suite Updates
The following test suites have been updated in `run_suite.py`:

1. **per-commit**: Added `test_audio_multimodal_expanded.py` (450s estimated)
2. **per-commit-2-gpu**: Added `test_vision_openai_server_tp2.py` (800s estimated)
3. **per-commit-4-gpu**: Added `test_vision_openai_server_tp4.py` (900s estimated)

### Execution Examples

```bash
# Run TP=2 tests
python3 -m unittest test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_stress_test_with_memory_monitoring

# Run TP=4 tests  
python3 -m unittest test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_large_batch_stress_test

# Run audio expansion tests
python3 -m unittest test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_audio_transcription_stress

# Run full suites
cd test/srt
python3 run_suite.py --suite per-commit-2-gpu  # Includes TP=2 tests
python3 run_suite.py --suite per-commit-4-gpu  # Includes TP=4 tests
```

## Addressing Issue Requirements

### ✅ TP-2/4 Configurations
- **Requirement**: "introduce test cases for TP-2/4 configurations"
- **Implementation**: Created dedicated test files for TP=2 and TP=4 with comprehensive test coverage
- **Integration**: Added to appropriate CI suites based on GPU requirements

### ✅ Stress Tests and Memory Leak Detection
- **Requirement**: "Stress tests are necessary, and simultaneously, it is crucial to investigate whether there are any memory leaks"
- **Implementation**: 
  - Multiple stress testing scenarios with varying concurrency levels
  - Comprehensive memory monitoring with leak detection
  - Configurable memory thresholds with detailed reporting
  - Both short-burst and sustained load testing

### ✅ Audio Expansion
- **Requirement**: "expand its scope to audio"
- **Implementation**:
  - Comprehensive audio processing test suite
  - Mixed audio/visual batch processing
  - Audio-specific memory monitoring
  - Audio quality and format robustness testing
  - TP=2 audio performance testing

### ✅ Enhanced CI Coverage
- **Requirement**: Improve overall multi-modal CI robustness
- **Implementation**:
  - Integration with existing test suite structure
  - Proper time estimates for test scheduling
  - Comprehensive coverage across different model types
  - Performance benchmarking and assertions

## Memory Leak Detection Strategy

### Thresholds
- **Single GPU**: 500MB increase threshold
- **TP=2**: 500MB increase threshold  
- **TP=4**: 800MB increase threshold (higher due to distributed memory)
- **Audio Processing**: 400MB increase threshold
- **Sustained Load**: 600MB increase threshold

### Monitoring Approach
1. **Baseline Measurement**: Record initial memory before test execution
2. **Continuous Sampling**: Monitor memory usage throughout test execution
3. **Peak Detection**: Track maximum memory usage during test
4. **Final Assessment**: Compare final memory to initial baseline
5. **Threshold Validation**: Assert memory increase stays within acceptable limits

### Leak Detection Features
- Garbage collection forcing between test iterations
- Memory sampling with configurable intervals
- Event correlation for debugging
- Detailed statistics reporting
- Automated threshold assertions

## Performance Benchmarking

### Throughput Metrics
- Images/second processing rates
- Audio files/second processing rates
- Mixed modality request handling rates
- Concurrent request completion rates

### Latency Metrics
- Average processing time per request
- Peak response times under load
- Sustained load performance over time

### Scalability Testing
- Variable concurrency level testing
- Resource utilization optimization
- TP scaling effectiveness validation

## Future Enhancements

### Potential Additions
1. **TP=8 Testing**: For larger scale deployments
2. **Pipeline Parallelism**: PP testing in combination with TP
3. **Model-Specific Optimizations**: Tailored tests for specific model architectures
4. **Cross-Modal Testing**: More complex multi-modal scenarios
5. **Performance Regression Detection**: Automated performance tracking

### Monitoring Improvements
1. **GPU Memory Detailed Tracking**: Per-GPU memory usage
2. **Network I/O Monitoring**: For distributed setups
3. **Disk I/O Tracking**: For model loading and caching
4. **Real-time Alerting**: For CI failure patterns

## Troubleshooting

### Common Issues
1. **Memory Threshold Exceeded**: Check for actual leaks vs. normal variance
2. **TP Configuration Failures**: Verify GPU availability and configuration
3. **Audio File Download Issues**: Check network connectivity and cache directory
4. **Timeout Issues**: Adjust timeout values for slower hardware

### Debug Information
- Memory statistics are printed to console for analysis
- Event logging available for audio processing correlation
- Detailed error messages with context information
- Performance metrics included in test output

## Conclusion

These enhancements significantly improve the robustness and coverage of multi-modal CI testing by:

1. **Addressing Scalability**: TP-2/4 configurations ensure the system works under distributed scenarios
2. **Ensuring Reliability**: Comprehensive stress testing and memory leak detection prevent regressions
3. **Expanding Coverage**: Audio testing ensures multi-modal capabilities beyond just vision
4. **Maintaining Quality**: Integrated CI execution ensures continuous validation

The implementation provides a solid foundation for ongoing multi-modal model development and deployment confidence.