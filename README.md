## Fused Sparse Tree (FST) Attention — Triton Kernel

This project implements Fused Sparse Tree (FST) Attention as a custom Triton kernel for NVIDIA GPUs. It accelerates attention by loading only shared ancestors and leaf nodes per query in a tree structure, optionally combining with a fully shared KV prefix. The repo also includes a simple correctness test against dense masked attention and a performance benchmark comparing with FlashAttention-2 (optional).

### Features
- FST attention forward kernel written in Triton
- Tree-aware sparse KV loading (ancestors + leaf self)
- Optional shared KV prefix support
- Quick correctness check vs dense masked attention
- Benchmark harness with Triton testing utilities; optional FlashAttention-2 comparison


### Project Layout
- `fst.py`: Triton kernel, PyTorch wrapper `fst_attention`, test function `test_op`, and benchmark entrypoint.
- `utils` module: Expected to provide tree utilities and constants `create_tree`, `create_fst_attention_kernel_inputs`, `create_full_attention_mask`, `DEPTH_MAPPING`, `MAX_ANCESTOR_MAPPING`. Make sure this module is on `PYTHONPATH`.

### Usage
Run the basic correctness tests and performance benchmark:
```bash
python fst.py
```
This will:
- Build and run the Triton FST kernel
- Validate output vs dense masked attention for two configurations (with and without shared KV prefix)
- Execute a performance sweep over sequence lengths and print timing data
- If FlashAttention-2 is installed, also report its timings for comparison

### API
- `fst_attention(q, k, v, ancestor_idx, ancestor_mask, leaf_idx, sm_scale, BLOCK_M=128, MAX_ANCESTORS=64)`
  - Inputs are standard Q/K/V tensors plus precomputed tree indices/masks; returns attention output shaped like `q`.
- `test_op(Z, H, N_CTX, D_HEAD, shared_kv_prefix=True, dtype=torch.float16)`
  - Generates synthetic data, builds tree structures, and checks numerical correctness vs dense masked attention.

### Notes
- Kernel assumes head dimension in `{16, 32, 64, 128}` and uses fp16 by default for data paths.
- The benchmark uses Triton testing utilities; results will vary by GPU, driver, and PyTorch/Triton versions.
- Ensure the `utils` module matches the expected API; otherwise, provide your own implementations for the tree construction helpers.



