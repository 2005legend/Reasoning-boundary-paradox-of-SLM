#!/usr/bin/env python3
"""
Script to add Cell 8: Quantized Model Loader Function to the notebook.
This implements Task 3.1: Load Qwen2.5 models with 4-bit NF4 quantization.
"""

import json

# Read the notebook
with open('rlvr_training_pipeline.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Create markdown cell
markdown_source = """## Cell 8: Quantized Model Loader Function

**Task 3.1**: Implement quantized model loader function to load Qwen2.5 models (0.5B and 1.5B) with 4-bit NF4 quantization, double quantization, and gradient checkpointing.

**Requirements:** 3.1, 3.2, 3.6

**Memory Optimization Strategy:**
- 4-bit NF4 quantization (~4x model size reduction)
- Double quantization for quantization constants
- Gradient checkpointing to reduce activation memory
- Must fit within T4 14GB VRAM constraint"""

new_cell_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": markdown_source.split('\n')
}

# Create code cell
code_source = '''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Optional
import gc

def load_quantized_model(
    model_size: str,
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads Qwen2.5-Instruct model with 4-bit NF4 quantization for memory efficiency.
    
    This function implements aggressive memory optimization to fit models within T4 GPU
    VRAM constraints (14GB available after system overhead).
    
    Memory Optimization Techniques:
    1. 4-bit NF4 quantization: Reduces model weights to ~4 bits per parameter (~4x compression)
    2. Double quantization: Quantizes the quantization constants themselves
    3. Gradient checkpointing: Trades computation for memory during backward pass
    
    Requirements:
    - 3.1: Load Qwen2.5-0.5B-Instruct and 1.5B-Instruct with 4-bit NF4 quantization
    - 3.2: Configure double quantization for additional memory savings
    - 3.6: Apply gradient checkpointing to reduce activation memory
    
    Args:
        model_size: Model size identifier ("0.5B", "1.5B", or "3B")
        device_map: Device mapping strategy ("auto" for automatic distribution)
        trust_remote_code: Whether to trust remote code from HuggingFace Hub
    
    Returns:
        Tuple of (model, tokenizer) with quantization applied
    
    Raises:
        ValueError: If model_size is invalid
        RuntimeError: If VRAM allocation fails
    
    Example:
        >>> model, tokenizer = load_quantized_model("0.5B")
        >>> print(f"Model loaded with {model.num_parameters()} parameters")
    """
    print("=" * 80)
    print("QUANTIZED MODEL LOADER")
    print("=" * 80)
    
    # Validate model size
    valid_sizes = ["0.5B", "1.5B", "3B"]
    if model_size not in valid_sizes:
        raise ValueError(
            f"Invalid model_size: '{model_size}'. "
            f"Must be one of {valid_sizes}. "
            f"\\n  → Use '0.5B' for Qwen2.5-0.5B-Instruct"
            f"\\n  → Use '1.5B' for Qwen2.5-1.5B-Instruct"
            f"\\n  → Use '3B' for Qwen2.5-3B-Instruct (experimental, high VRAM)"
        )
    
    # Map model size to HuggingFace model identifier
    model_id_map = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
        "3B": "Qwen/Qwen2.5-3B-Instruct"
    }
    model_id = model_id_map[model_size]
    
    print(f"\\n✓ Model: {model_id}")
    print(f"✓ Quantization: 4-bit NF4 with double quantization")
    print(f"✓ Compute type: bfloat16")
    print(f"✓ Device map: {device_map}")
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Display pre-load VRAM status
        vram_free_pre = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_reserved(0)) / (1024 ** 3)
        print(f"\\nVRAM before loading: {vram_free_pre:.2f} GB free")
    
    # Configure 4-bit quantization (Requirement 3.1, 3.2)
    # Reference: https://huggingface.co/docs/transformers/main/en/quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",              # Use NF4 (Normal Float 4) quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
        bnb_4bit_use_double_quant=True,         # Enable double quantization (Req 3.2)
    )
    
    print("\\n" + "=" * 80)
    print("LOADING MODEL (this may take 30-60 seconds)...")
    print("=" * 80)
    
    try:
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
        
        print("\\n✅ Model loaded successfully!")
        
        # Enable gradient checkpointing (Requirement 3.6)
        # This trades computation for memory by recomputing activations during backward pass
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
        
        # Load tokenizer
        print("\\n" + "=" * 80)
        print("LOADING TOKENIZER...")
        print("=" * 80)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✓ Set pad_token = eos_token")
        
        print("✅ Tokenizer loaded successfully!")
        
        # Display model information
        print("\\n" + "=" * 80)
        print("MODEL INFORMATION")
        print("=" * 80)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        # Display post-load VRAM status
        if torch.cuda.is_available():
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            vram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            vram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            vram_free_post = vram_total - vram_reserved
            
            print(f"\\nVRAM after loading:")
            print(f"  Total:     {vram_total:.2f} GB")
            print(f"  Reserved:  {vram_reserved:.2f} GB")
            print(f"  Allocated: {vram_allocated:.2f} GB")
            print(f"  Free:      {vram_free_post:.2f} GB")
            
            # Check VRAM constraint (Requirement 3.1)
            min_free_vram = 2.0  # Need at least 2GB free for training operations
            if vram_free_post < min_free_vram:
                print(f"\\n⚠️ WARNING: Low VRAM headroom!")
                print(f"   Free VRAM ({vram_free_post:.2f} GB) is below recommended minimum ({min_free_vram} GB)")
                print(f"   Training may encounter OOM errors. Consider:")
                print(f"     1. Using a smaller model size")
                print(f"     2. Reducing batch_size to 1 or 2")
                print(f"     3. Increasing gradient_accumulation_steps")
            else:
                print(f"\\n✅ VRAM check passed: {vram_free_post:.2f} GB free (above {min_free_vram} GB minimum)")
        
        # Display tokenizer vocabulary size
        print(f"\\nTokenizer vocabulary size: {len(tokenizer):,}")
        print(f"Model max length: {tokenizer.model_max_length:,}")
        
        print("\\n" + "=" * 80)
        print("MODEL LOADING COMPLETE")
        print("=" * 80)
        print("\\n✓ Model and tokenizer ready for training!\\n")
        
        return model, tokenizer
        
    except RuntimeError as e:
        # Requirement 3.6: Error handling for model loading failures
        print("\\n❌ ERROR: Model loading failed!")
        print("=" * 80)
        print(f"Error message: {str(e)}")
        
        # Check if it's a VRAM allocation error
        if "CUDA out of memory" in str(e) or "OutOfMemoryError" in str(e):
            print("\\n❌ CUDA Out of Memory Error")
            print("\\nThis error occurs when the GPU does not have enough VRAM to load the model.")
            
            if torch.cuda.is_available():
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                print(f"\\nCurrent VRAM status:")
                print(f"  Total VRAM: {vram_total:.2f} GB")
                print(f"  Reserved:   {vram_reserved:.2f} GB")
            
            print("\\nRecommended solutions:")
            print("  1. Use a smaller model size:")
            if model_size == "3B":
                print("     → Try '1.5B' instead of '3B'")
            elif model_size == "1.5B":
                print("     → Try '0.5B' instead of '1.5B'")
            print("  2. Restart the notebook runtime to clear memory:")
            print("     Runtime > Restart runtime")
            print("  3. Close other tabs/notebooks using GPU resources")
            print("  4. Use Colab Pro for more VRAM (recommended for 3B models)")
        
        print("\\n" + "=" * 80)
        raise RuntimeError(f"Failed to load {model_size} model: {str(e)}")
    
    except Exception as e:
        print(f"\\n❌ Unexpected error during model loading: {str(e)}")
        print("\\nIf this persists, please:")
        print("  1. Check internet connection")
        print("  2. Verify HuggingFace Hub access")
        print("  3. Restart the notebook runtime")
        raise

# Demo: Load a model (commented out by default to avoid automatic execution)
# Uncomment to test model loading
if __name__ == "__main__":
    # Example: Load 0.5B model
    print("To load a model, uncomment and run:")
    print("  model, tokenizer = load_quantized_model('0.5B')")
    print("\\nOr for 1.5B:")
    print("  model, tokenizer = load_quantized_model('1.5B')")
    
    # Uncomment to actually load:
    # model, tokenizer = load_quantized_model("0.5B")'''

new_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_source.split('\n')
}

# Add the cells to the notebook
notebook['cells'].append(new_cell_markdown)
notebook['cells'].append(new_cell_code)

# Write back to the notebook
with open('rlvr_training_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Successfully added Cell 8 (Quantized Model Loader) to the notebook!")
print(f"   Total cells now: {len(notebook['cells'])}")
