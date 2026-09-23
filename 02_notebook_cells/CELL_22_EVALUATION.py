# ============================================================================
# CELL 22: MODEL EVALUATION
# ============================================================================
# Evaluate the trained GRPO model on GSM8K test set

import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import Counter

print("=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)
print()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to your trained model in Google Drive (Colab)
MODEL_PATH = Path("/content/drive/MyDrive/RLVR_Research/checkpoints/exp2_n1_vanilla_0.5B/final_model")

# Alternative: Use checkpoint-800 instead
# MODEL_PATH = Path("/content/drive/MyDrive/RLVR_Research/checkpoints/exp2_n1_vanilla_0.5B/checkpoint-800")

# If running locally, use this instead:
# MODEL_PATH = Path("RLVR_Research-20260907T053709Z-1-001/RLVR_Research/checkpoints/exp2_n1_vanilla_0.5B/final_model")

# Evaluation settings
NUM_TEST_SAMPLES = 100  # Start with 100 samples (set to 1319 for full test set)
NUM_GENERATIONS = 4     # Generate 4 solutions per problem (Pass@k evaluation)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

print(f"Model path: {MODEL_PATH}")
print(f"Test samples: {NUM_TEST_SAMPLES}")
print(f"Generations per problem: {NUM_GENERATIONS}")
print()

# ==============================================================================
# STEP 1: LOAD MODEL AND TOKENIZER
# ==============================================================================
print("⏳ STEP 1/5: Loading trained model...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"   Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("   Make sure the model path is correct")
    raise

# ==============================================================================
# STEP 2: LOAD TEST DATASET
# ==============================================================================
print("\n⏳ STEP 2/5: Loading GSM8K test set...")

try:
    from datasets import load_dataset
    
    # Load GSM8K test set
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    
    # Format: "Question: ...\nAnswer: ..."
    test_problems = [item['question'] for item in gsm8k_test]
    test_answers = [item['answer'] for item in gsm8k_test]
    
    # Extract just the numeric answer (after ####)
    def extract_answer(answer_text: str) -> str:
        """Extract numeric answer from GSM8K format"""
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
        if match:
            return match.group(1).replace(',', '')
        return ""
    
    test_ground_truths = [extract_answer(ans) for ans in test_answers]
    
    # Use subset for evaluation
    if NUM_TEST_SAMPLES < len(test_problems):
        test_problems = test_problems[:NUM_TEST_SAMPLES]
        test_ground_truths = test_ground_truths[:NUM_TEST_SAMPLES]
    
    print(f"✅ Loaded {len(test_problems)} test problems")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    print("   Creating dummy test set...")
    test_problems = ["What is 2 + 2?"] * 10
    test_ground_truths = ["4"] * 10

# ==============================================================================
# STEP 3: DEFINE EVALUATION FUNCTIONS
# ==============================================================================
print("\n⏳ STEP 3/5: Setting up evaluation functions...")

def format_prompt(problem: str) -> str:
    """Format problem with XML reasoning template"""
    prompt = f"""Solve this math problem step by step. Show your reasoning in XML format.

Problem: {problem}

Provide your answer in this exact format:
<reasoning>
[Your step-by-step reasoning here]
</reasoning>
<answer>
[Your final numerical answer]
</answer>

Solution:"""
    return prompt

def generate_solutions(model, tokenizer, prompt: str, num_generations: int = 4) -> List[str]:
    """Generate multiple solutions for a single problem"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    solutions = []
    for _ in range(num_generations):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt
        if "Solution:" in solution:
            solution = solution.split("Solution:")[-1].strip()
        solutions.append(solution)
    
    return solutions

def extract_predicted_answer(text: str) -> str:
    """Extract answer from model output"""
    # Try XML format first
    match = re.search(r'<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>', text)
    if match:
        return match.group(1)
    
    # Try \boxed{} format
    match = re.search(r'\\boxed\{(-?\d+(?:\.\d+)?)\}', text)
    if match:
        return match.group(1)
    
    # Try finding last number in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""

def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    # Clean up both
    predicted = predicted.strip().replace(',', '')
    ground_truth = ground_truth.strip().replace(',', '')
    
    # Direct match
    if predicted == ground_truth:
        return True
    
    # Try numeric comparison
    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        return abs(pred_num - gt_num) < 0.01
    except:
        return False

print("✅ Evaluation functions ready")

# ==============================================================================
# STEP 4: RUN EVALUATION
# ==============================================================================
print("\n⏳ STEP 4/5: Running evaluation...")
print(f"   Generating {NUM_GENERATIONS} solutions per problem")
print(f"   This will take ~{NUM_TEST_SAMPLES * NUM_GENERATIONS * 10 // 60} minutes")
print()

results = []
pass_at_1 = []
pass_at_4 = []

for idx, (problem, ground_truth) in enumerate(tqdm(zip(test_problems, test_ground_truths), 
                                                      total=len(test_problems),
                                                      desc="Evaluating")):
    # Format prompt
    prompt = format_prompt(problem)
    
    # Generate solutions
    solutions = generate_solutions(model, tokenizer, prompt, NUM_GENERATIONS)
    
    # Extract answers
    predicted_answers = [extract_predicted_answer(sol) for sol in solutions]
    
    # Check correctness
    correct = [check_correctness(pred, ground_truth) for pred in predicted_answers]
    
    # Pass@1: At least 1 correct out of 4
    pass_at_1.append(1 if correct[0] else 0)
    
    # Pass@4: At least 1 correct out of 4
    pass_at_4.append(1 if any(correct) else 0)
    
    # Store result
    results.append({
        "problem_idx": idx,
        "ground_truth": ground_truth,
        "predictions": predicted_answers,
        "correct": correct,
        "pass@1": correct[0],
        "pass@4": any(correct)
    })
    
    # Progress update every 10 problems
    if (idx + 1) % 10 == 0:
        current_p1 = sum(pass_at_1) / len(pass_at_1) * 100
        current_p4 = sum(pass_at_4) / len(pass_at_4) * 100
        print(f"   Step {idx+1}/{NUM_TEST_SAMPLES}: Pass@1={current_p1:.1f}%, Pass@4={current_p4:.1f}%")

print("\n✅ Evaluation complete")

# ==============================================================================
# STEP 5: COMPUTE FINAL METRICS
# ==============================================================================
print("\n⏳ STEP 5/5: Computing final metrics...")

# Compute metrics
pass_at_1_accuracy = sum(pass_at_1) / len(pass_at_1) * 100
pass_at_4_accuracy = sum(pass_at_4) / len(pass_at_4) * 100

# Save results
results_dir = Path("/content/drive/MyDrive/RLVR_Research/results/exp2_n1_vanilla_0.5B")
results_dir.mkdir(parents=True, exist_ok=True)

evaluation_results = {
    "model": str(MODEL_PATH),
    "num_test_samples": NUM_TEST_SAMPLES,
    "num_generations": NUM_GENERATIONS,
    "pass_at_1": pass_at_1_accuracy,
    "pass_at_4": pass_at_4_accuracy,
    "total_correct_p1": sum(pass_at_1),
    "total_correct_p4": sum(pass_at_4),
    "temperature": TEMPERATURE,
    "max_new_tokens": MAX_NEW_TOKENS
}

# Save metrics
with open(results_dir / "evaluation_metrics.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

# Save detailed results
with open(results_dir / "evaluation_details.json", 'w') as f:
    json.dump(results, f, indent=2)

# ==============================================================================
# PRINT RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("📊 EVALUATION RESULTS")
print("=" * 80)
print()
print(f"Model: {MODEL_PATH.name}")
print(f"Test samples: {NUM_TEST_SAMPLES}")
print(f"Generations per problem: {NUM_GENERATIONS}")
print()
print(f"🎯 Pass@1 Accuracy: {pass_at_1_accuracy:.2f}%")
print(f"   ({sum(pass_at_1)}/{len(pass_at_1)} problems solved)")
print()
print(f"🎯 Pass@4 Accuracy: {pass_at_4_accuracy:.2f}%")
print(f"   ({sum(pass_at_4)}/{len(pass_at_4)} problems solved)")
print()
print(f"📈 Improvement from Pass@1 → Pass@4: {pass_at_4_accuracy - pass_at_1_accuracy:.2f}%")
print()
print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)
print(f"✅ Metrics: {results_dir / 'evaluation_metrics.json'}")
print(f"✅ Details: {results_dir / 'evaluation_details.json'}")
print("=" * 80)

# ==============================================================================
# SAMPLE OUTPUTS
# ==============================================================================
print("\n" + "=" * 80)
print("📝 SAMPLE OUTPUTS (First 3 problems)")
print("=" * 80)

for idx in range(min(3, len(results))):
    r = results[idx]
    print(f"\nProblem {idx+1}:")
    print(f"  Ground Truth: {r['ground_truth']}")
    print(f"  Pass@1: {'✅' if r['pass@1'] else '❌'}")
    print(f"  Pass@4: {'✅' if r['pass@4'] else '❌'}")
    print(f"  Predictions: {r['predictions']}")
    print(f"  Correct: {r['correct']}")

print("\n" + "=" * 80)
