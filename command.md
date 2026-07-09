# Command Cheat Sheet

Quick reference for running this fork's customized `live-vlm-webui` (YOLO-seg overlay, multi-backend model dropdown) on DGX Spark.

## Start the webui

```bash
cd /home/aath/live_vlm/live-vlm-webui
source /home/aath/live_vlm/.venv/bin/activate

live-vlm-webui --api-base http://localhost:8000/v1 --model qwen2.5-vl-7b \
  --trigger-mode yolo --yolo-model /home/aath/live_vlm/models/yolo11n-seg.pt
```

Add `&` at the end (or run in a separate terminal/tmux pane) to keep using the shell. Requires the vLLM backend to be running first (see below) - the webui itself will still start without it, but VLM calls will fail until a backend is reachable.

Access at `https://localhost:8090` (or `https://<LAN-IP>:8090` from another device on the network).

## Stop the webui

```bash
live-vlm-webui-stop
```

Finds and gracefully stops the running server (force-kills if it doesn't exit within a couple seconds).

**To also free the VLM backend's memory** (stops any running vLLM Docker container + unloads any loaded Ollama model - useful since vLLM alone can hold ~70-90GB on this unified-memory system):

```bash
live-vlm-webui-stop --free-memory
```

Leave this flag off if you want the backend to stay warm for an instant reconnect next time.

## vLLM backend container

```bash
# Start (first time load takes ~1-2 min: weight load + CUDA graph capture)
docker start vllm-qwen2_5-vl-7b

# Check health
docker inspect --format='{{.State.Health.Status}}' vllm-qwen2_5-vl-7b

# Watch it come up
docker logs -f vllm-qwen2_5-vl-7b

# Stop (or just use `live-vlm-webui-stop --free-memory` instead)
docker stop vllm-qwen2_5-vl-7b
```

## Ollama backend (alternative, lighter model)

```bash
# List pulled models
curl http://localhost:11434/api/tags

# Check what's currently loaded in memory
ollama ps

# Manually unload a model
ollama stop gemma3:4b
```

## Switching models in the browser

The model dropdown lists models from every detected local backend (vLLM, Ollama, SGLang) at once, grouped by backend. Picking a model auto-switches the "API Base" field to match it - no manual backend switching needed. Use the refresh icon next to the dropdown to re-scan if you started a backend after the page was already loaded.

## Quick diagnostics

```bash
# Is the webui running, and on what port?
ps aux | grep "[l]ive-vlm-webui"
ss -ltnp | grep 8090

# Is vLLM reachable?
curl -s http://localhost:8000/v1/models

# Is Ollama reachable?
curl -s http://localhost:11434/api/tags

# Memory/GPU usage (unified memory on GB10 - GPU procs show in both)
free -h
nvidia-smi
```
