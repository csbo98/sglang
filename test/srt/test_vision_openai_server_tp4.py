"""
Test suite for multi-modal models with Tensor Parallelism (TP=4) configurations.
This addresses the issue requirement for TP-2/4 test cases in multi-modal CI.

Usage:
python3 -m unittest test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_single_image_chat_completion
python3 -m unittest test_vision_openai_server_tp4.TestQwen2VLServerTP4.test_large_batch_stress_test
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


class AdvancedMemoryMonitor:
    """Advanced memory monitoring with GPU memory tracking."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = 0
        self.monitoring = False
        self.memory_samples = []
        self.gpu_memory_samples = []
        
    def start_monitoring(self):
        """Start monitoring system and GPU memory usage."""
        self.monitoring = True
        self.initial_memory = psutil.virtual_memory().used
        self.peak_memory = self.initial_memory
        self.memory_samples = []
        self.gpu_memory_samples = []
        
        def monitor():
            while self.monitoring:
                # System memory
                current_memory = psutil.virtual_memory().used
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                
                # GPU memory (if available)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated()
                        self.gpu_memory_samples.append(gpu_memory)
                except ImportError:
                    pass
                
                time.sleep(0.5)  # Sample every 500ms
                
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return comprehensive memory statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
        final_memory = psutil.virtual_memory().used
        stats = {
            'initial_memory_mb': self.initial_memory / (1024 * 1024),
            'final_memory_mb': final_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'memory_increase_mb': (final_memory - self.initial_memory) / (1024 * 1024),
            'peak_increase_mb': (self.peak_memory - self.initial_memory) / (1024 * 1024),
            'samples_count': len(self.memory_samples)
        }
        
        # Add GPU memory stats if available
        if self.gpu_memory_samples:
            stats['gpu_peak_memory_mb'] = max(self.gpu_memory_samples) / (1024 * 1024)
            stats['gpu_final_memory_mb'] = self.gpu_memory_samples[-1] / (1024 * 1024)
        
        return stats


class TestQwen2VLServerTP4(TestOpenAIVisionServer):
    """Test Qwen2-VL with TP=4 configuration."""
    
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
                "--tp-size", "4",  # Enable TP=4
                "--mem-fraction-static", "0.5",
                "--cuda-graph-max-bs", "16",
                "--disable-radix-cache",  # For more predictable memory usage
            ],
        )
        cls.base_url += "/v1"

    def test_single_image_chat_completion_tp4(self):
        """Test basic single image functionality with TP=4."""
        super().test_single_image_chat_completion()

    def test_multi_images_chat_completion_tp4(self):
        """Test multi-image functionality with TP=4."""
        super().test_multi_images_chat_completion()

    def test_video_images_chat_completion_tp4(self):
        """Test video processing with TP=4."""
        super().test_video_images_chat_completion()

    def test_large_batch_stress_test(self):
        """Large batch stress test with memory leak detection for TP=4."""
        monitor = AdvancedMemoryMonitor()
        monitor.start_monitoring()
        
        try:
            client = openai.Client(api_key=self.api_key, base_url=self.base_url)
            
            # Run larger concurrent requests to leverage TP=4
            num_concurrent_requests = 16  # Higher than TP=2 tests
            num_iterations = 3
            
            def make_request(iteration, request_id):
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": IMAGE_MAN_IRONING_URL if request_id % 2 == 0 else IMAGE_SGL_LOGO_URL},
                                },
                                {
                                    "type": "text",
                                    "text": f"Analyze this image in detail (TP=4, iter {iteration}, req {request_id}).",
                                },
                            ],
                        },
                    ],
                    temperature=0,
                    max_tokens=100,  # Longer responses to stress the system
                )
                return response.choices[0].message.content
            
            # Execute large batch stress test
            all_responses = []
            for iteration in range(num_iterations):
                print(f"Running TP=4 stress test iteration {iteration + 1}/{num_iterations}")
                
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
                time.sleep(2)  # Longer pause for TP=4
            
            # Verify all requests completed successfully
            expected_total = num_concurrent_requests * num_iterations
            self.assertEqual(len(all_responses), expected_total)
            
        finally:
            memory_stats = monitor.stop_monitoring()
            
            # Print comprehensive memory statistics
            print(f"\n--- Memory Statistics for TP=4 Large Batch Stress Test ---")
            print(f"Initial Memory: {memory_stats['initial_memory_mb']:.2f} MB")
            print(f"Final Memory: {memory_stats['final_memory_mb']:.2f} MB")
            print(f"Peak Memory: {memory_stats['peak_memory_mb']:.2f} MB")
            print(f"Memory Increase: {memory_stats['memory_increase_mb']:.2f} MB")
            print(f"Peak Increase: {memory_stats['peak_increase_mb']:.2f} MB")
            print(f"Samples Collected: {memory_stats['samples_count']}")
            
            if 'gpu_peak_memory_mb' in memory_stats:
                print(f"GPU Peak Memory: {memory_stats['gpu_peak_memory_mb']:.2f} MB")
                print(f"GPU Final Memory: {memory_stats['gpu_final_memory_mb']:.2f} MB")
            
            # Assert reasonable memory usage for TP=4 (higher threshold)
            self.assertLess(
                memory_stats['memory_increase_mb'], 
                800,  # Higher threshold for TP=4
                f"Memory increase of {memory_stats['memory_increase_mb']:.2f} MB exceeds threshold, possible memory leak"
            )

    def test_high_concurrency_mixed_modalities_tp4(self):
        """Test high concurrency with mixed modalities on TP=4."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        def single_image_request(request_id):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": IMAGE_SGL_LOGO_URL}},
                        {"type": "text", "text": f"Describe this logo (req {request_id})."}
                    ]
                }],
                temperature=0,
                max_tokens=50
            )
        
        def multi_image_request(request_id):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": IMAGE_MAN_IRONING_URL}, "modalities": "multi-images"},
                        {"type": "image_url", "image_url": {"url": IMAGE_SGL_LOGO_URL}, "modalities": "multi-images"},
                        {"type": "text", "text": f"Compare these images (req {request_id})."}
                    ]
                }],
                temperature=0,
                max_tokens=80
            )
        
        def text_only_request(request_id):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user", 
                    "content": f"Hello, this is text-only request {request_id}. How are you?"
                }],
                temperature=0,
                max_tokens=30
            )
        
        # Run high concurrency mixed requests
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            
            # Submit varied request types
            for i in range(4):
                futures.append(executor.submit(single_image_request, f"single_{i}"))
                futures.append(executor.submit(multi_image_request, f"multi_{i}"))
                futures.append(executor.submit(text_only_request, f"text_{i}"))
            
            # Collect all results
            responses = []
            for future in as_completed(futures):
                response = future.result()
                responses.append(response.choices[0].message.content)
                self.assertIsInstance(response.choices[0].message.content, str)
                self.assertGreater(len(response.choices[0].message.content), 0)
        
        self.assertEqual(len(responses), 12)

    def test_throughput_benchmark_tp4(self):
        """Benchmark throughput capabilities of TP=4 setup."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Large batch of images for throughput testing
        image_urls = [IMAGE_MAN_IRONING_URL, IMAGE_SGL_LOGO_URL] * 25  # 50 total requests
        
        start_time = time.time()
        
        def process_image(url, index):
            return client.chat.completions.create(
                model="default",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": f"Describe image {index} concisely."}
                    ]
                }],
                temperature=0,
                max_tokens=30
            )
        
        # Process all images with high concurrency
        with ThreadPoolExecutor(max_workers=20) as executor:
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
        
        print(f"\n--- TP=4 Throughput Benchmark Results ---")
        print(f"Processed {len(image_urls)} images in {processing_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")
        print(f"Average time per image: {processing_time/len(image_urls):.3f} seconds")
        
        # Verify all requests completed successfully
        self.assertEqual(len(results), len(image_urls))
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        
        # Assert minimum throughput for TP=4 (should be better than single GPU)
        self.assertGreater(throughput, 1.0, "TP=4 throughput should exceed 1 image/second")


class TestQwen2_5_VLServerTP4(TestOpenAIVisionServer):
    """Test Qwen2.5-VL with TP=4 configuration."""
    
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
                "--tp-size", "4",  # Enable TP=4
                "--mem-fraction-static", "0.5",
                "--cuda-graph-max-bs", "16",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion_tp4(self):
        """Test video processing with TP=4."""
        self._test_video_chat_completion()

    def test_sustained_load_test_tp4(self):
        """Test sustained load over extended period with TP=4."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        monitor = AdvancedMemoryMonitor()
        monitor.start_monitoring()
        
        try:
            # Sustained load test - moderate concurrency over longer duration
            num_concurrent = 8
            num_rounds = 10
            requests_per_round = 6
            
            def make_sustained_request(round_num, req_id):
                return client.chat.completions.create(
                    model="default",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": IMAGE_MAN_IRONING_URL}},
                            {"type": "text", "text": f"Sustained test round {round_num}, request {req_id}. Describe briefly."}
                        ]
                    }],
                    temperature=0,
                    max_tokens=40
                )
            
            total_requests = 0
            start_time = time.time()
            
            for round_num in range(num_rounds):
                print(f"Sustained load test round {round_num + 1}/{num_rounds}")
                
                with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                    futures = [
                        executor.submit(make_sustained_request, round_num, req_id)
                        for req_id in range(requests_per_round)
                    ]
                    
                    for future in as_completed(futures):
                        response = future.result()
                        self.assertIsInstance(response.choices[0].message.content, str)
                        self.assertGreater(len(response.choices[0].message.content), 0)
                        total_requests += 1
                
                # Brief pause between rounds
                time.sleep(1)
            
            end_time = time.time()
            total_time = end_time - start_time
            average_throughput = total_requests / total_time
            
            print(f"\n--- TP=4 Sustained Load Test Results ---")
            print(f"Total requests: {total_requests}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average throughput: {average_throughput:.2f} requests/second")
            
            self.assertEqual(total_requests, num_rounds * requests_per_round)
            
        finally:
            memory_stats = monitor.stop_monitoring()
            
            print(f"\n--- Sustained Load Memory Statistics ---")
            print(f"Memory increase: {memory_stats['memory_increase_mb']:.2f} MB")
            print(f"Peak increase: {memory_stats['peak_increase_mb']:.2f} MB")
            
            # Memory should remain stable during sustained load
            self.assertLess(
                memory_stats['memory_increase_mb'], 
                600, 
                f"Memory increase during sustained load exceeds threshold: {memory_stats['memory_increase_mb']:.2f} MB"
            )


if __name__ == "__main__":
    del TestOpenAIVisionServer
    unittest.main()