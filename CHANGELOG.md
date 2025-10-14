# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-01-01

### Added
- Initial implementation of Fused Sparse Tree (FST) Attention
- Triton kernel for tree-aware sparse attention computation
- PyTorch wrapper function for easy integration
- Tree utility functions for hierarchical structure generation
- Correctness testing against dense masked attention
- Performance benchmarking with Triton testing framework
- Optional FlashAttention-2 integration for comparison
- Comprehensive documentation and setup instructions
- Support for shared KV prefix configurations
- Configurable tree depth and branching factors

### Features
- Tree-aware sparse KV loading (ancestors + leaf self-attention)
- Optional shared KV prefix support for efficient caching
- GPU-optimized memory access patterns
- Support for head dimensions: 16, 32, 64, 128
- FP16 precision with configurable data types
- Sequence length scaling from 1K to 64K tokens
