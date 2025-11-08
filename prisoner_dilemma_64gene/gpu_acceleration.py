"""
🚀 GPU ACCELERATION FOR ULTIMATE ECHO SIMULATION
================================================

This module adds GPU support using CuPy (CUDA) or PyTorch for:
1. Distance calculations (neighbor finding)
2. Batch genetic operations (crossover, mutation)
3. Payoff matrix calculations

GPU is most beneficial for:
- Large populations (>1000 agents)
- High vision range (requires many distance calculations)
- Batch reproduction (many genetic operations)

Requirements:
- NVIDIA GPU with CUDA support
- CuPy: pip install cupy-cuda11x (or cupy-cuda12x)
OR
- PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Try to import GPU libraries
GPU_AVAILABLE = False
GPU_BACKEND = None

try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_BACKEND = "cupy"
    print("✅ CuPy GPU backend loaded")
except ImportError:
    try:
        import torch
        if torch.cuda.is_available():
            GPU_AVAILABLE = True
            GPU_BACKEND = "pytorch"
            print("✅ PyTorch GPU backend loaded")
        else:
            print("⚠️ PyTorch found but no CUDA GPU available")
    except ImportError:
        print("⚠️ No GPU backend available (install cupy or torch)")


class GPUAccelerator:
    """
    GPU acceleration wrapper for Echo simulation operations.
    
    Automatically falls back to CPU if GPU not available.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.backend = GPU_BACKEND if self.use_gpu else "cpu"
        
        if self.use_gpu:
            print(f"🚀 GPU acceleration enabled ({self.backend})")
        else:
            print("💻 Running on CPU")
    
    def manhattan_distances_batch(
        self, 
        positions: np.ndarray, 
        target_pos: Tuple[int, int],
        grid_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Calculate Manhattan distances from target to all positions.
        
        GPU-accelerated for large arrays.
        
        Args:
            positions: Nx2 array of (x, y) positions
            target_pos: (x, y) target position
            grid_size: (width, height) for wrapping
            
        Returns:
            Array of Manhattan distances
        """
        if self.use_gpu and len(positions) > 100 and self.backend == "cupy":
            # CuPy implementation
            positions_gpu = cp.array(positions)
            target_gpu = cp.array(target_pos)
            grid_gpu = cp.array(grid_size)
            
            # Calculate differences
            dx = cp.abs(positions_gpu[:, 0] - target_gpu[0])
            dy = cp.abs(positions_gpu[:, 1] - target_gpu[1])
            
            # Handle wrapping
            dx = cp.minimum(dx, grid_gpu[0] - dx)
            dy = cp.minimum(dy, grid_gpu[1] - dy)
            
            # Manhattan distance
            distances = dx + dy
            
            # Move back to CPU
            return cp.asnumpy(distances)
        
        elif self.use_gpu and len(positions) > 100 and self.backend == "pytorch":
            # PyTorch implementation
            positions_gpu = torch.tensor(positions, device='cuda')
            target_gpu = torch.tensor(target_pos, device='cuda')
            grid_gpu = torch.tensor(grid_size, device='cuda')
            
            # Calculate differences
            dx = torch.abs(positions_gpu[:, 0] - target_gpu[0])
            dy = torch.abs(positions_gpu[:, 1] - target_gpu[1])
            
            # Handle wrapping
            dx = torch.minimum(dx, grid_gpu[0] - dx)
            dy = torch.minimum(dy, grid_gpu[1] - dy)
            
            # Manhattan distance
            distances = dx + dy
            
            # Move back to CPU
            return distances.cpu().numpy()
        
        else:
            # CPU fallback
            dx = np.abs(positions[:, 0] - target_pos[0])
            dy = np.abs(positions[:, 1] - target_pos[1])
            
            # Handle wrapping
            dx = np.minimum(dx, grid_size[0] - dx)
            dy = np.minimum(dy, grid_size[1] - dy)
            
            return dx + dy
    
    def batch_crossover(
        self,
        parents1: np.ndarray,
        parents2: np.ndarray,
        chromosome_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch genetic crossover operation.
        
        GPU-accelerated for large batches.
        
        Args:
            parents1: Nx21 array of parent 1 chromosomes
            parents2: Nx21 array of parent 2 chromosomes
            chromosome_length: Length of chromosome (21)
            
        Returns:
            Tuple of (children1, children2) arrays
        """
        n_pairs = len(parents1)
        
        if self.use_gpu and n_pairs > 50 and self.backend == "cupy":
            # CuPy implementation
            parents1_gpu = cp.array(parents1)
            parents2_gpu = cp.array(parents2)
            
            # Random crossover points
            crossover_points = cp.random.randint(1, chromosome_length, n_pairs)
            
            # Create masks
            children1 = cp.zeros_like(parents1_gpu)
            children2 = cp.zeros_like(parents2_gpu)
            
            for i in range(n_pairs):
                point = int(crossover_points[i])
                children1[i, :point] = parents1_gpu[i, :point]
                children1[i, point:] = parents2_gpu[i, point:]
                children2[i, :point] = parents2_gpu[i, :point]
                children2[i, point:] = parents1_gpu[i, point:]
            
            return cp.asnumpy(children1), cp.asnumpy(children2)
        
        elif self.use_gpu and n_pairs > 50 and self.backend == "pytorch":
            # PyTorch implementation
            parents1_gpu = torch.tensor(parents1, device='cuda')
            parents2_gpu = torch.tensor(parents2, device='cuda')
            
            # Random crossover points
            crossover_points = torch.randint(1, chromosome_length, (n_pairs,), device='cuda')
            
            # Create children
            children1 = torch.zeros_like(parents1_gpu)
            children2 = torch.zeros_like(parents2_gpu)
            
            for i in range(n_pairs):
                point = int(crossover_points[i])
                children1[i, :point] = parents1_gpu[i, :point]
                children1[i, point:] = parents2_gpu[i, point:]
                children2[i, :point] = parents2_gpu[i, :point]
                children2[i, point:] = parents1_gpu[i, point:]
            
            return children1.cpu().numpy(), children2.cpu().numpy()
        
        else:
            # CPU fallback
            children1 = np.zeros_like(parents1)
            children2 = np.zeros_like(parents2)
            
            for i in range(n_pairs):
                point = np.random.randint(1, chromosome_length)
                children1[i, :point] = parents1[i, :point]
                children1[i, point:] = parents2[i, point:]
                children2[i, :point] = parents2[i, :point]
                children2[i, point:] = parents1[i, point:]
            
            return children1, children2
    
    def batch_mutation(
        self,
        chromosomes: np.ndarray,
        mutation_rate: float
    ) -> np.ndarray:
        """
        Batch mutation operation.
        
        GPU-accelerated for large batches.
        
        Args:
            chromosomes: NxL array of chromosomes
            mutation_rate: Probability of bit flip
            
        Returns:
            Mutated chromosomes
        """
        if self.use_gpu and len(chromosomes) > 50 and self.backend == "cupy":
            # CuPy implementation
            chromosomes_gpu = cp.array(chromosomes)
            mutation_mask = cp.random.random(chromosomes_gpu.shape) < mutation_rate
            mutated = cp.where(mutation_mask, 1 - chromosomes_gpu, chromosomes_gpu)
            return cp.asnumpy(mutated)
        
        elif self.use_gpu and len(chromosomes) > 50 and self.backend == "pytorch":
            # PyTorch implementation
            chromosomes_gpu = torch.tensor(chromosomes, device='cuda', dtype=torch.float32)
            mutation_mask = torch.rand(chromosomes_gpu.shape, device='cuda') < mutation_rate
            mutated = torch.where(mutation_mask, 1 - chromosomes_gpu, chromosomes_gpu)
            return mutated.cpu().numpy().astype(int)
        
        else:
            # CPU fallback
            mutated = chromosomes.copy()
            mutation_mask = np.random.random(chromosomes.shape) < mutation_rate
            mutated[mutation_mask] = 1 - mutated[mutation_mask]
            return mutated
    
    def get_stats(self) -> Dict:
        """Get GPU statistics."""
        if not self.use_gpu:
            return {"backend": "cpu", "available": False}
        
        stats = {
            "backend": self.backend,
            "available": True
        }
        
        if self.backend == "cupy":
            try:
                mempool = cp.get_default_memory_pool()
                stats["memory_used"] = mempool.used_bytes() / 1024**2  # MB
                stats["memory_total"] = mempool.total_bytes() / 1024**2  # MB
            except:
                pass
        
        elif self.backend == "pytorch":
            try:
                stats["memory_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
                stats["memory_reserved"] = torch.cuda.memory_reserved() / 1024**2  # MB
                stats["device_name"] = torch.cuda.get_device_name(0)
            except:
                pass
        
        return stats


# Global GPU accelerator instance
_gpu_accelerator = None

def get_gpu_accelerator(use_gpu: bool = True) -> GPUAccelerator:
    """Get or create global GPU accelerator instance."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(use_gpu=use_gpu)
    return _gpu_accelerator


def check_gpu_available() -> Dict:
    """
    Check if GPU is available and return info.
    
    Returns:
        Dict with GPU availability and info
    """
    info = {
        "available": GPU_AVAILABLE,
        "backend": GPU_BACKEND
    }
    
    if GPU_AVAILABLE:
        if GPU_BACKEND == "cupy":
            try:
                info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
                info["device_count"] = cp.cuda.runtime.getDeviceCount()
                info["device_name"] = cp.cuda.Device(0).name
            except:
                pass
        
        elif GPU_BACKEND == "pytorch":
            try:
                info["cuda_available"] = torch.cuda.is_available()
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda
            except:
                pass
    
    return info


if __name__ == "__main__":
    print("🚀 GPU ACCELERATION MODULE")
    print("=" * 60)
    
    # Check GPU availability
    info = check_gpu_available()
    print(f"\nGPU Available: {info['available']}")
    if info['available']:
        print(f"Backend: {info['backend']}")
        for key, value in info.items():
            if key not in ['available', 'backend']:
                print(f"  {key}: {value}")
    
    # Test GPU accelerator
    print("\n" + "=" * 60)
    print("Testing GPU Accelerator...")
    print("=" * 60)
    
    gpu = GPUAccelerator(use_gpu=True)
    
    # Test 1: Distance calculations
    print("\n📏 Test 1: Manhattan Distance Calculation")
    positions = np.random.randint(0, 50, (1000, 2))
    target = (25, 25)
    grid_size = (50, 50)
    
    import time
    start = time.time()
    distances = gpu.manhattan_distances_batch(positions, target, grid_size)
    elapsed = time.time() - start
    
    print(f"  Calculated {len(positions)} distances in {elapsed*1000:.2f}ms")
    print(f"  Min distance: {distances.min():.1f}")
    print(f"  Max distance: {distances.max():.1f}")
    print(f"  Avg distance: {distances.mean():.1f}")
    
    # Test 2: Batch crossover
    print("\n🧬 Test 2: Batch Genetic Crossover")
    n_pairs = 100
    parents1 = np.random.randint(0, 2, (n_pairs, 21))
    parents2 = np.random.randint(0, 2, (n_pairs, 21))
    
    start = time.time()
    children1, children2 = gpu.batch_crossover(parents1, parents2, 21)
    elapsed = time.time() - start
    
    print(f"  Crossed {n_pairs} parent pairs in {elapsed*1000:.2f}ms")
    print(f"  Children shape: {children1.shape}")
    
    # Test 3: Batch mutation
    print("\n🔬 Test 3: Batch Mutation")
    chromosomes = np.random.randint(0, 2, (100, 21))
    mutation_rate = 0.01
    
    start = time.time()
    mutated = gpu.batch_mutation(chromosomes, mutation_rate)
    elapsed = time.time() - start
    
    print(f"  Mutated {len(chromosomes)} chromosomes in {elapsed*1000:.2f}ms")
    print(f"  Mutation rate: {mutation_rate}")
    
    # GPU stats
    print("\n📊 GPU Statistics:")
    stats = gpu.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ GPU acceleration module loaded successfully!")
