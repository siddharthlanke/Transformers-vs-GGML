# Transformers-vs-GGML
This repo compares text generation using Hugging Face Transformers and GGML-quantized models, demonstrating both approaches on a free Google Colab T4 GPU.

---

This repo demonstrates two approaches to text generation:  
- **Hugging Face Transformers** (`codellama/CodeLlama-7b-hf`) for GPU-optimized inference.  
- **GGML models** (`TheBloke/Llama-2-13B-chat-GGML`) via `ctransformers` for efficient CPU/GPU hybrid workflows.  


---

### Key Differences

| Feature                | Hugging Face Transformers (Cell 1)       | GGML + ctransformers (Cell 2)           |
|------------------------|------------------------------------------|------------------------------------------|
| **Model Format**       | PyTorch (16/32-bit)                     | GGML (4/8-bit quantized)                |
| **Library**            | `transformers`                          | `ctransformers`                         |
| **Hardware**           | GPU-optimized                           | CPU/GPU hybrid                          |
| **Memory Usage**       | High (~14GB for 7B)                     | Low (~4GB for 13B quantized)            |
| **Use Case**           | High-performance GPUs                   | Resource-constrained devices            |
| **Model Size**         | 7B (13.5GB)                             | 13B (8GB quantized)                     |

---

### **What is GGML?**
- A tensor library for efficient inference on CPUs and GPUs.
- Uses **quantization** to reduce model size (e.g., 4-bit or 8-bit weights).
- Enables running large models on consumer hardware.

### **Why Use ctransformers?**
- Python bindings for GGML models.
- No CUDA/PyTorch dependencies.
- Supports partial GPU offloading for faster inference.

---

### **ctransformers Config Parameters**

| Parameter             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `max_new_tokens`      | Maximum number of tokens to generate (controls output length).              |
| `repetition_penalty`  | Penalizes repeated phrases (1.0 = no penalty, >1.0 reduces repetition).     |
| `temperature`         | Controls randomness (0.1 = deterministic, 1.0 = creative).                  |
| `stream`              | Stream output token-by-token (not used here since `stream=False`).          |
| `gpu_layers`          | Number of layers to offload to GPU (130 for 13B, 110 for 7B).               |

---

### **Model Comparison**

#### 1. **`TheBloke/Llama-2-7B-GGML`**
- **Size**: 7 billion parameters.
- **Use Case**: Lightweight tasks on low-resource devices.
- **Quantization**: 4-bit or 8-bit options.
- **Speed**: ~20 tokens/sec on CPU.

#### 2. **`TheBloke/Llama-2-7B-chat-GGML`**
- **Specialization**: Fine-tuned for conversational tasks.
- **Use Case**: Chatbots, dialogue systems.
- **Example Prompt**: "How do I reset my password?"

#### 3. **`TheBloke/Llama-2-13B-GGML`**
- **Size**: 13 billion parameters.
- **Use Case**: Complex text generation tasks.
- **Hardware**: Requires more RAM (16GB+ recommended).

#### 4. **`TheBloke/Llama-2-13B-chat-GGML`**
- **Specialization**: High-quality conversational responses.
- **Use Case**: Advanced chatbots, customer support.
- **Example Prompt**: "Explain quantum computing in simple terms."

---

### **When to Choose Which Model?**

| Scenario                      | Recommended Model                          |
|-------------------------------|--------------------------------------------|
| Low RAM (8GB) + CPU           | `Llama-2-7B-GGML`                          |
| Conversational tasks          | `Llama-2-7B-chat-GGML`                     |
| High-quality text generation  | `Llama-2-13B-GGML`                         |
| Advanced chatbots             | `Llama-2-13B-chat-GGML`                    |
| Quick prototyping             | `Llama-2-7B-GGML` (fastest)                |

---
