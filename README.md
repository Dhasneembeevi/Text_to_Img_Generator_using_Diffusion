# Stable Diffusion Image Generation in Google Colab

## Overview
This Google Colab project utilizes the `diffusers` library from Hugging Face to generate high-quality images from text prompts using the **Stable Diffusion XL Base 1.0** model. The model runs on a CUDA-enabled GPU for efficient processing.

## Features
- Uses **Stable Diffusion XL Base 1.0** for high-resolution image generation.
- Implements **fp16 precision** for optimized performance.
- Allows the use of **negative prompts** to refine image outputs.
- Outputs images directly within the notebook using `matplotlib`.

## Prerequisites
Ensure that your Google Colab environment supports GPU acceleration. Go to:
```
Runtime > Change runtime type > Hardware accelerator > GPU
```

## Installation
Before running the script, install the required dependencies:
```python
!pip install diffusers torch matplotlib
```

## Usage
### 1. Import Required Libraries
```python
from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
```

### 2. Load the Model
```python
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")  # Move model to GPU
```

### 3. Generate an Image
```python
prompt = "A vibrant sunset over the city skyline with silhouetted buildings."
negative_prompt = "Avoid including any water elements in the scene."
images = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
```

### 4. Display the Image
```python
plt.imshow(images)
plt.axis('off')
plt.show()
```

## Notes
- **Negative prompts** help in avoiding unwanted elements in the generated image.
- The model runs best on **NVIDIA GPUs** due to `fp16` optimization.
- Ensure that `torch` is installed with CUDA support for better performance.


