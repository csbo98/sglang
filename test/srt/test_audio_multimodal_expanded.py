"""
Expanded audio testing suite for multi-modal CI.
This addresses the issue requirement to expand VLM CI scope to audio beyond just vision.

Usage:
python3 -m unittest test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_audio_transcription_stress
python3 -m unittest test_audio_multimodal_expanded.TestAudioProcessingExpanded.test_mixed_audio_visual_batch
"""

import gc
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

from test_vision_openai_server_common import *

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Additional audio test resources
AUDIO_MUSIC_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/classical_music_10s.mp3"
AUDIO_NOISE_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/white_noise_5s.mp3"
AUDIO_MULTILINGUAL_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/multilingual_speech.mp3"


class AudioMemoryMonitor:
    """Specialized memory monitor for audio processing tests."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = 0
        self.monitoring = False
        self.memory_samples = []
        self.audio_processing_events = []
        
    def start_monitoring(self):
        """Start monitoring memory usage during audio processing."""
        self.monitoring = True
        self.initial_memory = psutil.virtual_memory().used
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        self.audio_processing_events = []
        
        def monitor():
            while self.monitoring:
                current_memory = psutil.virtual_memory().used
                self.memory_samples.append({
                    'timestamp': time.time(),
                    'memory': current_memory
                })
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.3)  # More frequent sampling for audio
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def log_audio_event(self, event_type, details=None):
        """Log audio processing events for correlation with memory usage."""
        self.audio_processing_events.append({
            'timestamp': time.time(),
            'event': event_type,
            'details': details or {}
        })
        
    def stop_monitoring(self):
        """Stop monitoring and return audio-specific memory statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
        final_memory = psutil.virtual_memory().used
        return {
            'initial_memory_mb': self.initial_memory / (1024 * 1024),
            'final_memory_mb': final_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'memory_increase_mb': (final_memory - self.initial_memory) / (1024 * 1024),
            'peak_increase_mb': (self.peak_memory - self.initial_memory) / (1024 * 1024),
            'samples_count': len(self.memory_samples),
            'audio_events_count': len(self.audio_processing_events)
        }


class TestAudioProcessingExpanded(TestOpenAIVisionServer):
    """Expanded audio processing tests for multi-modal CI."""
    
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"  # Model with good audio support
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "8",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_transcription_stress(self):
        """Stress test for audio transcription with memory monitoring."""
        monitor = AudioMemoryMonitor()
        monitor.start_monitoring()
        
        try:
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Test multiple audio files concurrently
            audio_urls = [
                AUDIO_TRUMP_SPEECH_URL,
                AUDIO_BIRD_SONG_URL,
                AUDIO_TRUMP_SPEECH_URL,  # Repeat for stress testing
                AUDIO_BIRD_SONG_URL,
            ]
            
            def process_audio(url, request_id):
                monitor.log_audio_event('start_processing', {'url': url, 'request_id': request_id})
                
                audio_file_path = self.get_or_download_file(url)
                messages = self.prepare_audio_messages(
                    f"Transcribe this audio file (request {request_id}).", 
                    audio_file_path
                )
                
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=0,
                    max_tokens=150,
                    stream=False,
                )
                
                monitor.log_audio_event('finish_processing', {'request_id': request_id})
                return response.choices[0].message.content
            
            # Process audio files concurrently
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(process_audio, url, i)
                    for i, url in enumerate(audio_urls)
                ]
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    self.assertIsInstance(result, str)
                    self.assertGreater(len(result), 0)
            
            # Verify all audio processing completed
            self.assertEqual(len(results), len(audio_urls))
            
        finally:
            memory_stats = monitor.stop_monitoring()
            
            print(f"\n--- Audio Transcription Stress Test Results ---")
            print(f"Processed {len(audio_urls)} audio files")
            print(f"Memory increase: {memory_stats['memory_increase_mb']:.2f} MB")
            print(f"Peak increase: {memory_stats['peak_increase_mb']:.2f} MB")
            print(f"Audio events logged: {memory_stats['audio_events_count']}")
            
            # Assert reasonable memory usage for audio processing
            self.assertLess(
                memory_stats['memory_increase_mb'], 
                400, 
                f"Audio processing memory increase exceeds threshold: {memory_stats['memory_increase_mb']:.2f} MB"
            )

    def test_mixed_audio_visual_batch(self):
        """Test concurrent audio and visual processing in mixed batches."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        def audio_transcription_request(audio_url, request_id):
            audio_file_path = self.get_or_download_file(audio_url)
            messages = self.prepare_audio_messages(
                f"Briefly transcribe this audio (req {request_id}).", 
                audio_file_path
            )
            return client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0,
                max_tokens=80
            )
        
        def image_description_request(image_url, request_id):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": f"Briefly describe this image (req {request_id})."}
                    ]
                }],
                temperature=0,
                max_tokens=60
            )
        
        def text_only_request(request_id):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user", 
                    "content": f"Hello, this is text request {request_id}. Respond briefly."
                }],
                temperature=0,
                max_tokens=30
            )
        
        # Create mixed batch of requests
        with ThreadPoolExecutor(max_workers=9) as executor:
            futures = []
            
            # Submit 3 audio, 3 visual, 3 text requests
            for i in range(3):
                futures.append(executor.submit(
                    audio_transcription_request, 
                    AUDIO_TRUMP_SPEECH_URL if i % 2 == 0 else AUDIO_BIRD_SONG_URL, 
                    f"audio_{i}"
                ))
                futures.append(executor.submit(
                    image_description_request, 
                    IMAGE_MAN_IRONING_URL if i % 2 == 0 else IMAGE_SGL_LOGO_URL, 
                    f"image_{i}"
                ))
                futures.append(executor.submit(text_only_request, f"text_{i}"))
            
            # Collect all results
            responses = []
            for future in as_completed(futures):
                response = future.result()
                content = response.choices[0].message.content
                responses.append(content)
                self.assertIsInstance(content, str)
                self.assertGreater(len(content), 0)
        
        # Verify all 9 requests completed
        self.assertEqual(len(responses), 9)
        print(f"\n--- Mixed Audio/Visual Batch Test ---")
        print(f"Successfully processed {len(responses)} mixed modality requests")

    def test_audio_quality_analysis(self):
        """Test audio quality analysis capabilities."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Test different types of audio content
        test_cases = [
            {
                'url': AUDIO_TRUMP_SPEECH_URL,
                'prompt': "Analyze the audio quality and describe the speaker's voice characteristics.",
                'expected_keywords': ['voice', 'speech', 'quality', 'clear', 'audio']
            },
            {
                'url': AUDIO_BIRD_SONG_URL,
                'prompt': "Describe the audio content and identify what type of sounds you hear.",
                'expected_keywords': ['bird', 'sound', 'nature', 'chirp', 'song']
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            with self.subTest(test_case=i):
                audio_file_path = self.get_or_download_file(test_case['url'])
                messages = self.prepare_audio_messages(test_case['prompt'], audio_file_path)
                
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=0,
                    max_tokens=120
                )
                
                content = response.choices[0].message.content.lower()
                self.assertIsInstance(content, str)
                self.assertGreater(len(content), 20)
                
                # Check for expected keywords (at least one should be present)
                found_keywords = [kw for kw in test_case['expected_keywords'] if kw in content]
                self.assertGreater(
                    len(found_keywords), 0, 
                    f"No expected keywords found in response: {content}"
                )
                
                print(f"Audio quality test {i+1}: Found keywords {found_keywords}")

    def test_audio_format_robustness(self):
        """Test robustness with different audio formats and edge cases."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Test with the available audio files
        audio_files = [AUDIO_TRUMP_SPEECH_URL, AUDIO_BIRD_SONG_URL]
        
        def process_with_different_prompts(url, prompt_type):
            audio_file_path = self.get_or_download_file(url)
            
            prompts = {
                'transcribe': "Please transcribe this audio.",
                'summarize': "Summarize what you hear in this audio.",
                'analyze': "Analyze the content and context of this audio.",
                'identify': "Identify the type of audio and main elements."
            }
            
            messages = self.prepare_audio_messages(prompts[prompt_type], audio_file_path)
            
            return client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0,
                max_tokens=100
            )
        
        # Test different prompt types with different audio files
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for audio_url in audio_files:
                for prompt_type in ['transcribe', 'summarize', 'analyze', 'identify']:
                    futures.append(executor.submit(
                        process_with_different_prompts, 
                        audio_url, 
                        prompt_type
                    ))
            
            results = []
            for future in as_completed(futures):
                response = future.result()
                content = response.choices[0].message.content
                results.append(content)
                self.assertIsInstance(content, str)
                self.assertGreater(len(content), 0)
        
        # Verify all combinations worked
        expected_combinations = len(audio_files) * 4  # 4 prompt types
        self.assertEqual(len(results), expected_combinations)
        
        print(f"\n--- Audio Format Robustness Test ---")
        print(f"Successfully tested {expected_combinations} audio processing combinations")

    def test_audio_memory_leak_detection(self):
        """Specific test for audio processing memory leaks."""
        monitor = AudioMemoryMonitor()
        monitor.start_monitoring()
        
        try:
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Repeated audio processing to detect leaks
            num_iterations = 8
            
            def process_audio_iteration(iteration):
                audio_file_path = self.get_or_download_file(AUDIO_TRUMP_SPEECH_URL)
                messages = self.prepare_audio_messages(
                    f"Transcribe briefly (iteration {iteration}).", 
                    audio_file_path
                )
                
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=0,
                    max_tokens=50
                )
                
                # Force cleanup
                gc.collect()
                return response.choices[0].message.content
            
            # Process audio files sequentially to detect memory accumulation
            results = []
            for i in range(num_iterations):
                monitor.log_audio_event('iteration_start', {'iteration': i})
                result = process_audio_iteration(i)
                results.append(result)
                monitor.log_audio_event('iteration_end', {'iteration': i})
                
                # Brief pause between iterations
                time.sleep(0.5)
            
            # Verify all iterations completed
            self.assertEqual(len(results), num_iterations)
            
        finally:
            memory_stats = monitor.stop_monitoring()
            
            print(f"\n--- Audio Memory Leak Detection Results ---")
            print(f"Completed {num_iterations} audio processing iterations")
            print(f"Memory increase: {memory_stats['memory_increase_mb']:.2f} MB")
            print(f"Peak increase: {memory_stats['peak_increase_mb']:.2f} MB")
            print(f"Average memory per iteration: {memory_stats['memory_increase_mb']/num_iterations:.2f} MB")
            
            # Assert no significant memory leak (allow for some normal variance)
            self.assertLess(
                memory_stats['memory_increase_mb'], 
                200,  # 200MB threshold for 8 iterations
                f"Potential audio memory leak detected: {memory_stats['memory_increase_mb']:.2f} MB increase"
            )

    def test_concurrent_audio_processing_scalability(self):
        """Test scalability of concurrent audio processing."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Test increasing levels of concurrency
        concurrency_levels = [2, 4, 6]
        
        for concurrency in concurrency_levels:
            with self.subTest(concurrency=concurrency):
                start_time = time.time()
                
                def process_concurrent_audio(request_id):
                    audio_file_path = self.get_or_download_file(
                        AUDIO_TRUMP_SPEECH_URL if request_id % 2 == 0 else AUDIO_BIRD_SONG_URL
                    )
                    messages = self.prepare_audio_messages(
                        f"Brief transcription (concurrent {request_id}).", 
                        audio_file_path
                    )
                    
                    return client.chat.completions.create(
                        model="default",
                        messages=messages,
                        temperature=0,
                        max_tokens=40
                    )
                
                # Process with current concurrency level
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(process_concurrent_audio, i)
                        for i in range(concurrency)
                    ]
                    
                    results = []
                    for future in as_completed(futures):
                        response = future.result()
                        results.append(response.choices[0].message.content)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Verify all requests completed
                self.assertEqual(len(results), concurrency)
                
                # All results should be valid
                for result in results:
                    self.assertIsInstance(result, str)
                    self.assertGreater(len(result), 0)
                
                print(f"Concurrency {concurrency}: {processing_time:.2f}s for {concurrency} requests")


class TestAudioProcessingTP2(TestOpenAIVisionServer):
    """Audio processing tests with TP=2 configuration."""
    
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size", "2",  # Enable TP=2 for audio
                "--mem-fraction-static", "0.6",
                "--cuda-graph-max-bs", "8",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_processing_tp2_performance(self):
        """Test audio processing performance with TP=2."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Process multiple audio files to test TP=2 performance
        audio_urls = [AUDIO_TRUMP_SPEECH_URL, AUDIO_BIRD_SONG_URL] * 4  # 8 total
        
        start_time = time.time()
        
        def process_audio_tp2(url, index):
            audio_file_path = self.get_or_download_file(url)
            messages = self.prepare_audio_messages(
                f"Transcribe this audio file {index} briefly.", 
                audio_file_path
            )
            
            return client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0,
                max_tokens=60
            )
        
        # Process with high concurrency to leverage TP=2
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_audio_tp2, url, i)
                for i, url in enumerate(audio_urls)
            ]
            
            results = []
            for future in as_completed(futures):
                response = future.result()
                results.append(response.choices[0].message.content)
        
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = len(audio_urls) / processing_time
        
        print(f"\n--- Audio TP=2 Performance Test ---")
        print(f"Processed {len(audio_urls)} audio files in {processing_time:.2f} seconds")
        print(f"Audio throughput with TP=2: {throughput:.2f} files/second")
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), len(audio_urls))
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)


if __name__ == "__main__":
    del TestOpenAIVisionServer
    unittest.main()