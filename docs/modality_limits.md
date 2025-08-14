# Per-Modality Input Limits in SGLang

## Overview

SGLang now supports configurable per-modality input limits for multimodal inference pipelines. This feature allows you to cap the maximum number of inputs per modality (images, videos, audio) per prompt, providing better control over resource usage and preventing out-of-memory (OOM) errors.

## Motivation

In multimodal inference pipelines, a single prompt can contain multiple input items of various modalities. Without proper limits, this can lead to:

- **Excessive memory usage**: Large high-resolution media files can quickly exhaust GPU memory
- **Slow preprocessing**: Processing too many inputs can significantly slow down inference
- **Out-of-memory errors**: Uncontrolled input counts can cause OOM crashes
- **Unpredictable resource usage**: Different workloads have different needs

## Features

- **Flexible configuration**: Set limits via CLI arguments or JSON configuration
- **Per-modality control**: Independent limits for images, videos, and audio
- **Override capability**: Individual modality limits override JSON configuration
- **Clear error messages**: Informative errors when limits are exceeded
- **Zero-overhead when disabled**: No performance impact when limits are not configured

## Configuration Options

### 1. Individual Modality Limits

Use specific CLI arguments for each modality:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --enable-multimodal \
    --max-images-per-prompt 5 \
    --max-videos-per-prompt 1 \
    --max-audios-per-prompt 3
```

### 2. JSON Configuration

Use a JSON string to configure all limits at once:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --enable-multimodal \
    --limit-mm-per-prompt '{"image": 10, "video": 2, "audio": 5}'
```

### 3. Combined Configuration

Individual limits override JSON configuration:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --enable-multimodal \
    --limit-mm-per-prompt '{"image": 10, "video": 2, "audio": 5}' \
    --max-images-per-prompt 20  # Overrides the JSON value for images
```

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--limit-mm-per-prompt` | JSON string | JSON object specifying per-modality limits |
| `--max-images-per-prompt` | int | Maximum number of images per prompt |
| `--max-videos-per-prompt` | int | Maximum number of videos per prompt |
| `--max-audios-per-prompt` | int | Maximum number of audio inputs per prompt |

## Use Cases

### Production API with Memory Constraints

Prevent OOM by limiting to single image processing:

```bash
--max-images-per-prompt 1 --max-videos-per-prompt 0
```

### Document Analysis Service

Allow multiple document pages but block video/audio:

```bash
--max-images-per-prompt 10 --max-videos-per-prompt 0 --max-audios-per-prompt 0
```

### Video Summarization Service

Process one video at a time to manage GPU memory:

```bash
--max-images-per-prompt 0 --max-videos-per-prompt 1 --max-audios-per-prompt 0
```

### Multimodal Chatbot

Balanced limits for general use:

```bash
--limit-mm-per-prompt '{"image": 5, "video": 1, "audio": 3}'
```

## Error Handling

When a request exceeds configured limits, a clear error message is returned:

```
ValueError: Number of images (10) exceeds the maximum limit of 5 images per prompt.
Please reduce the number of images or adjust the limit using --max-images-per-prompt
or --limit-mm-per-prompt.
```

## API Usage

When making requests to an SGLang server with modality limits configured, ensure your requests comply with the limits:

```python
import requests

# This request will succeed if image limit >= 3
response = requests.post("http://localhost:30000/generate", json={
    "text": "Compare these images",
    "image_data": ["image1.jpg", "image2.jpg", "image3.jpg"],
})

# This request will fail if image limit < 5
response = requests.post("http://localhost:30000/generate", json={
    "text": "Analyze all these images",
    "image_data": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"],
})
```

## Implementation Details

The modality limit system consists of:

1. **ModalityLimitConfig**: Configuration dataclass for storing limits
2. **ModalityLimitValidator**: Validates requests against configured limits
3. **ServerArgs extensions**: New CLI arguments and parsing logic
4. **TokenizerManager integration**: Validation during request processing

The validation occurs early in the request pipeline, before any expensive preprocessing, ensuring fast rejection of invalid requests.

## Testing

Run the test suite to verify the modality limits functionality:

```bash
python -m pytest python/sglang/test/test_modality_limits.py -v
```

## Examples

See `examples/runtime/multimodal_limits_example.py` for a comprehensive demonstration:

```bash
python examples/runtime/multimodal_limits_example.py --show-all
```

## Compatibility

- Compatible with all multimodal models supported by SGLang
- No breaking changes to existing APIs
- Backward compatible - existing deployments continue to work without limits

## Performance

- **Zero overhead**: No performance impact when limits are not configured
- **Early validation**: Requests are validated before expensive preprocessing
- **Efficient counting**: Optimized item counting for nested list structures

## Future Enhancements

Potential future improvements:

- Per-user or per-API-key limits
- Dynamic limit adjustment based on available memory
- Soft limits with queueing
- Detailed metrics on rejected requests
- Support for custom modality types