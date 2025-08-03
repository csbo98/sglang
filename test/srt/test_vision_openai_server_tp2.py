"""
Test suite for multi-modal models with Tensor Parallelism (TP=2) configurations.
This addresses the issue requirement for TP-2/4 test cases in multi-modal CI.

Usage:
python3 -m unittest test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_single_image_chat_completion
python3 -m unittest test_vision_openai_server_tp2.TestQwen2VLServerTP2.test_stress_test_with_memory_monitoring
"""

import gc
import psutil
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


class MemoryMonitor:
    """Monitor memory usage for leak detection."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = 0
        self.monitoring = False
        self.memory_samples = []
        
    def start_monitoring(self):
        """Start monitoring memory usage."""
        self.monitoring = True
        self.initial_memory = psutil.virtual_memory().used
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        
        def monitor():
            while self.monitoring:
                current_memory = psutil.virtual_memory().used
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.5)  # Sample every 500ms
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return memory statistics."""
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
            'samples_count': len(self.memory_samples)
        }


class TestQwen2VLServerTP2(TestOpenAIVisionServer):
    """Test Qwen2-VL with TP=2 configuration."""
    
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tp-size", "2",  # Enable TP=2
                "--mem-fraction-static", "0.4",
                "--cuda-graph-max-bs", "8",
                "--disable-radix-cache",  # For more predictable memory usage
            ],
        )
        cls.base_url += "/v1"

    def test_single_image_chat_completion_tp2(self):
        """Test basic single image functionality with TP=2."""
        super().test_single_image_chat_completion()

    def test_multi_images_chat_completion_tp2(self):
        """Test multi-image functionality with TP=2."""
        super().test_multi_images_chat_completion()

    def test_video_images_chat_completion_tp2(self):
        """Test video processing with TP=2."""
        super().test_video_images_chat_completion()

    def test_mixed_batch_tp2(self):
        """Test mixed batch processing with TP=2."""
        super().test_mixed_batch()

    def test_stress_test_with_memory_monitoring(self):
        """Stress test with memory leak detection for TP=2."""
        monitor = MemoryMonitor()
        monitor.start_monitoring()
        
        try:
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Run multiple concurrent requests to stress test
            num_concurrent_requests = 8
            num_iterations = 5
            
            def make_request(iteration, request_id):
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                                },
                                {
                                    "type": "text",
                                    "text": f"Describe this image briefly (iteration {iteration}, request {request_id}).",
                                },
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=50,  # Keep responses short for stress testing
                )
                return response.choices[0].message.content
            
            # Execute stress test
            all_responses = []
            for iteration in range(num_iterations):
                with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
                    futures = [
                        executor.submit(make_request, iteration, req_id)
                        for req_id in range(num_concurrent_requests)
                    ]
                    
                    iteration_responses = []
                    for future in as_completed(futures):
                        response = future.result()
                        iteration_responses.append(response)
                        self.assertIsInstance(response, str)
                        self.assertGreater(len(response), 0)
                    
                    all_responses.extend(iteration_responses)
                
                # Force garbage collection between iterations
                gc.collect()
                time.sleep(1)  # Brief pause between iterations
            
            # Verify all requests completed successfully
            self.assertEqual(len(all_responses), num_concurrent_requests * num_iterations)
            
        finally:
            memory_stats = monitor.stop_monitoring()
            
            # Print memory statistics for analysis
            print(f"\n--- Memory Statistics for TP=2 Stress Test ---")
            print(f"Initial Memory: {memory_stats['initial_memory_mb']:.2f} MB")
            print(f"Final Memory: {memory_stats['final_memory_mb']:.2f} MB")
            print(f"Peak Memory: {memory_stats['peak_memory_mb']:.2f} MB")
            print(f"Memory Increase: {memory_stats['memory_increase_mb']:.2f} MB")
            print(f"Peak Increase: {memory_stats['peak_increase_mb']:.2f} MB")
            print(f"Samples Collected: {memory_stats['samples_count']}")
            
            # Assert reasonable memory usage (no major leaks)
            # Allow up to 500MB increase as reasonable for the stress test
            self.assertLess(
                memory_stats['memory_increase_mb'], 
                500, 
                f"Memory increase of {memory_stats['memory_increase_mb']:.2f} MB exceeds threshold, possible memory leak"
            )

    def test_concurrent_multimodal_requests_tp2(self):
        """Test concurrent requests with different modalities on TP=2."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        def image_request():
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": IMAGE_SGL_LOGO_URL}},
                        {"type": "text", "text": "What do you see?"}
                    ]
                }],
                temperature=0,
                max_tokens=30
            )
        
        def text_only_request():
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user", 
                    "content": "Hello, how are you today?"
                }],
                temperature=0,
                max_tokens=20
            )
        
        # Run mixed requests concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            # Submit 3 image requests and 3 text-only requests
            for _ in range(3):
                futures.append(executor.submit(image_request))
                futures.append(executor.submit(text_only_request))
            
            # Collect all results
            responses = []
            for future in as_completed(futures):
                response = future.result()
                responses.append(response.choices[0].message.content)
                self.assertIsInstance(response.choices[0].message.content, str)
                self.assertGreater(len(response.choices[0].message.content), 0)
        
        self.assertEqual(len(responses), 6)


class TestQwen2_5_VLServerTP2(TestOpenAIVisionServer):
    """Test Qwen2.5-VL with TP=2 configuration."""
    
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--tp-size", "2",  # Enable TP=2
                "--mem-fraction-static", "0.4",
                "--cuda-graph-max-bs", "8",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion_tp2(self):
        """Test video processing with TP=2."""
        self._test_video_chat_completion()

    def test_high_throughput_image_processing_tp2(self):
        """Test high throughput image processing with TP=2."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Process multiple images rapidly
        image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL] * 10  # 20 total requests
        
        start_time = time.time()
        
        def process_image(url, index):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": f"Describe image {index} in one word."}
                    ]
                }],
                temperature=0,
                max_tokens=5
            )
        
        # Process all images concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_image, url, i) 
                for i, url in enumerate(image_urls)
            ]
            
            results = []
            for future in as_completed(futures):
                response = future.result()
                results.append(response.choices[0].message.content)
        
        end_time = time.time()
        processing_time = end_time - start_time
        throughput = len(image_urls) / processing_time
        
        print(f"\n--- TP=2 Throughput Test Results ---")
        print(f"Processed {len(image_urls)} images in {processing_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), len(image_urls))
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)


class TestMinicpmoServerTP2(TestOpenAIVisionServer):
    """Test MiniCPM-o with TP=2 for audio capabilities."""
    
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
                "--tp-size", "2",  # Enable TP=2
                "--mem-fraction-static", "0.7",
                "--cuda-graph-max-bs", "6",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion_tp2(self):
        """Test audio processing with TP=2."""
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()

    def test_mixed_audio_visual_tp2(self):
        """Test concurrent audio and visual processing with TP=2."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        def audio_request():
            audio_file_path = self.get_or_download_file(AUDIO_TRUMP_SPEECH_URL)
            messages = self.prepare_audio_messages(
                "Transcribe this briefly.", 
                audio_file_path
            )
            return client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=0,
                max_tokens=50
            )
        
        def visual_request():
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": IMAGE_MAN_IRONING_URL}},
                        {"type": "text", "text": "Describe this image briefly."}
                    ]
                }],
                temperature=0,
                max_tokens=50
            )
        
        # Run mixed audio/visual requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(audio_request))
                futures.append(executor.submit(visual_request))
            
            responses = []
            for future in as_completed(futures):
                response = future.result()
                responses.append(response.choices[0].message.content)
        
        # Verify all responses
        self.assertEqual(len(responses), 4)
        for response in responses:
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)


if __name__ == "__main__":
    del TestOpenAIVisionServer
    unittest.main()