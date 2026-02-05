---
name: modal-ml-inference
description: Modal GPU deployment for ML model inference. Use when deploying diffusion models, video generation models, ComfyUI workflows, or custom ML pipelines with high-performance serverless GPU scaling and API endpoints.
---

# Modal ML Inference

Serverless GPU deployment for ML models with automatic scaling and performance optimization.

## When to Apply

- Deploying diffusion models (Stable Diffusion, LTX-Video, Mochi)
- Running ComfyUI workflows as APIs
- Video/image generation model inference
- High-throughput batch processing of ML models

## Critical Rules

**GPU Memory Snapshots**: Enable for faster cold starts with pre-loaded models

```python
# WRONG - Model loads on every cold start
@app.cls(gpu="H100")
class Inference:
    @modal.enter()
    def load_model(self):
        self.model = load_model()

# RIGHT - Snapshot captures GPU state with loaded model
@app.cls(
    gpu="H100",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True}
)
class Inference:
    @modal.enter(snap=True)
    def load_model(self):
        self.model = load_model()
        self.model.to("cuda")
```

**Volume Mounting**: Cache model weights to avoid repeated downloads

```python
# WRONG - Downloads model every time
self.model = DiffusionPipeline.from_pretrained("model-name")

# RIGHT - Cache in persistent volume
model_vol = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.cls(volumes={"/models": model_vol})
class Inference:
    @modal.enter()
    def load_model(self):
        # HF_HUB_CACHE points to volume mount
        self.model = DiffusionPipeline.from_pretrained("model-name")
```

**Concurrent Scaling**: Configure for your workload pattern

```python
# For ComfyUI workflows (5 concurrent per container)
@app.cls(
    scaledown_window=300,  # 5min keepalive
    gpu="L40S"
)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    pass

# For high-throughput batch inference
@app.cls(scaledown_window=15*60)  # 15min keepalive
@modal.concurrent(max_inputs=32, target_inputs=25)
class BatchInference:
    pass
```

## Key Patterns

### Diffusion Model API

```python
@app.cls(
    image=image,
    volumes={"/models": model_vol, "/outputs": output_vol},
    gpu="H100",
    timeout=10*60,
    scaledown_window=15*60
)
class DiffusionAPI:
    @modal.enter()
    def load_model(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "model-name", torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

    @modal.method()
    def generate(self, prompt: str, **kwargs):
        frames = self.pipe(prompt=prompt, **kwargs).frames[0]
        output_path = f"/outputs/{uuid.uuid4()}.mp4"
        export_to_video(frames, output_path)
        output_vol.commit()
        return output_path
```

### FastAPI Inference Endpoint

```python
@app.function(image=web_image)
@modal.fastapi_endpoint(method="POST", docs=True)
def inference_api(request: GenerationRequest):
    result = DiffusionAPI().generate.remote(
        prompt=request.prompt,
        num_inference_steps=request.steps
    )
    return {"filename": result}
```

### ComfyUI Workflow Service

```python
@app.cls(
    scaledown_window=300,
    gpu="L40S",
    volumes={"/cache": vol}
)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_server(self):
        subprocess.run(
            f"comfy launch --background -- --port {self.port}",
            shell=True, check=True
        )

    @modal.method()
    def run_workflow(self, workflow_path: str):
        subprocess.run(
            f"comfy run --workflow {workflow_path} --wait --timeout 1200",
            shell=True, check=True
        )
        return self.get_output_image()
```

### Efficient Container Images

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04")
    .entrypoint([])  # Remove verbose logging
    .apt_install("git")
    .uv_pip_install(
        "torch==2.7.1",
        "transformers~=4.53.0",
        "diffusers",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    .env({"HF_HUB_CACHE": "/models"})
)
```

## Common Mistakes

- **No volume caching** — Models download repeatedly, wasting time and bandwidth
- **Wrong scaledown_window** — Too short causes frequent cold starts, too long wastes money
- **Incorrect concurrency** — Single-input functions can't utilize full GPU capacity
- **Missing GPU snapshots** — Cold start times remain high for large models
- **Inadequate timeouts** — Video generation models need 10+ minute timeouts
- **No error handling** — Long-running inference needs retry configuration
