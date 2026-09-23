"""
Local Model Evaluation Script
Evaluates the trained GRPO model using local files
"""

import torch
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

print("=" * 80)
print("📊 LOCAL MODEL EVALUATION")
print("=" * 80)
print()

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ==============================================================================

# Path to your downloaded model
MODEL_PATH = Path(r"RLVR_Research-20260907T053709Z-1-001\RLVR_Research\checkpoints\exp2_n1_vanilla_0.5B\final_model")

# Path to GSM8K test set (if you have it locally)
# If not, we'll create a small test set from your training data
GSM8K_LOCAL_PATH = None  # Set to your local path if you have it

# Evaluation settings
NUM_TEST_SAMPLES = 20   # Start with 20 for quick test
NUM_GENERATIONS = 4     # Generate 4 solutions per problem
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
    from peft import PeftModel, PeftConfig
    
    # Check if path exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    
    print(f"   Reading PEFT config...")
    peft_config = PeftConfig.from_pretrained(str(MODEL_PATH))
    base_model_name = peft_config.base_model_name_or_path
    
    print(f"   Base model: {base_model_name}")
    print(f"   Loading base model (this may take a minute)...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"   Loading LoRA adapter...")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        str(MODEL_PATH),
        torch_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    device = next(model.parameters()).device
    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    print(f"   Dtype: {next(model.parameters()).dtype}")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("\nTroubleshooting:")
    print(f"  1. Check model path exists: {MODEL_PATH}")
    print(f"  2. Check you have GPU/CUDA available: {torch.cuda.is_available()}")
    print(f"  3. Try running on CPU (slower): device_map='cpu'")
    raise

# ==============================================================================
# STEP 2: LOAD TEST DATASET
# ==============================================================================
print("\n⏳ STEP 2/5: Loading test dataset...")

test_problems = []
test_ground_truths = []

# Option 1: Load from local GSM8K if available
if GSM8K_LOCAL_PATH and Path(GSM8K_LOCAL_PATH).exists():
    print(f"   Loading from local path: {GSM8K_LOCAL_PATH}")
    try:
        with open(GSM8K_LOCAL_PATH, 'r') as f:
            data = json.load(f)
        test_problems = [item['question'] for item in data]
        
        def extract_answer(answer_text):
            match = re.search(r'####\s*(-?\d+(?:[,\.\d]*)?)', answer_text)
            return match.group(1).replace(',', '') if match else ""
        
        test_ground_truths = [extract_answer(item['answer']) for item in data]
    except Exception as e:
        print(f"   ⚠️  Failed to load local dataset: {e}")

# Option 2: Try downloading from HuggingFace
if not test_problems:
    print("   Attempting to download GSM8K from HuggingFace...")
    try:
        from datasets import load_dataset
        gsm8k_test = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        
        test_problems = [item['question'] for item in gsm8k_test]
        test_answers = [item['answer'] for item in gsm8k_test]
        
        def extract_answer(answer_text):
            match = re.search(r'####\s*(-?\d+(?:[,\.\d]*)?)', answer_text)
            return match.group(1).replace(',', '') if match else ""
        
        test_ground_truths = [extract_answer(ans) for ans in test_answers]
        print(f"   ✅ Downloaded GSM8K from HuggingFace")
    except Exception as e:
        print(f"   ⚠️  Failed to download: {e}")

# Option 3: Use sample problems for testing
if not test_problems:
    print("   Using sample test problems...")
    test_problems = [
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
    ]
    test_ground_truths = ["18", "3", "70000", "540", "20"]

# Use subset
test_problems = test_problems[:NUM_TEST_SAMPLES]
test_ground_truths = test_ground_truths[:NUM_TEST_SAMPLES]

print(f"✅ Loaded {len(test_problems)} test problems")

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
        if "Solution:" in solution:
            solution = solution.split("Solution:")[-1].strip()
        solutions.append(solution)
    
    return solutions

def extract_predicted_answer(text: str) -> str:
    """Extract answer from model output"""
    # Try XML format
    match = re.search(r'<answer>\s*(-?\d+(?:[,\.\d]*)?)\s*</answer>', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Try \boxed{}
    match = re.search(r'\\boxed\{(-?\d+(?:[,\.\d]*)?)\}', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try "answer is" or "answer:"
    match = re.search(r'(?:answer is|answer:)\s*(-?\d+(?:[,\.\d]*)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Try last number
    numbers = re.findall(r'-?\d+(?:[,\.\d]*)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""

def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth"""
    predicted = predicted.strip().replace(',', '').replace(' ', '')
    ground_truth = ground_truth.strip().replace(',', '').replace(' ', '')
    
    if predicted == ground_truth:
        return True
    
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
print(f"   Estimated time: ~{NUM_TEST_SAMPLES * NUM_GENERATIONS * 3 // 60} minutes")
print()

results = []
pass_at_1 = []
pass_at_4 = []

for idx, (problem, ground_truth) in enumerate(tqdm(zip(test_problems, test_ground_truths), 
                                                      total=len(test_problems),
                                                      desc="Evaluating")):
    prompt = format_prompt(problem)
    solutions = generate_solutions(model, tokenizer, prompt, NUM_GENERATIONS)
    predicted_answers = [extract_predicted_answer(sol) for sol in solutions]
    correct = [check_correctness(pred, ground_truth) for pred in predicted_answers]
    
    pass_at_1.append(1 if correct[0] else 0)
    pass_at_4.append(1 if any(correct) else 0)
    
    results.append({
        "problem_idx": idx,
        "problem": problem,
        "ground_truth": ground_truth,
        "predictions": predicted_answers,
        "solutions": solutions,
        "correct": correct,
        "pass@1": correct[0],
        "pass@4": any(correct)
    })
    
    if (idx + 1) % 5 == 0 or idx == len(test_problems) - 1:
        current_p1 = sum(pass_at_1) / len(pass_at_1) * 100
        current_p4 = sum(pass_at_4) / len(pass_at_4) * 100
        print(f"   [{idx+1}/{NUM_TEST_SAMPLES}] Pass@1={current_p1:.1f}%, Pass@4={current_p4:.1f}%")

print("\n✅ Evaluation complete")

# ==============================================================================
# STEP 5: COMPUTE FINAL METRICS
# ==============================================================================
print("\n⏳ STEP 5/5: Computing final metrics...")

pass_at_1_accuracy = sum(pass_at_1) / len(pass_at_1) * 100
pass_at_4_accuracy = sum(pass_at_4) / len(pass_at_4) * 100

# Save results
results_dir = Path("evaluation_results")
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

with open(results_dir / "evaluation_metrics.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

with open(results_dir / "evaluation_details.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Results saved to: {results_dir.absolute()}")

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
print(f"   ({sum(pass_at_1)}/{len(pass_at_1)} problems solved on first try)")
print()
print(f"🎯 Pass@4 Accuracy: {pass_at_4_accuracy:.2f}%")
print(f"   ({sum(pass_at_4)}/{len(pass_at_4)} problems solved in 4 tries)")
print()
print(f"📈 Improvement: Pass@1 → Pass@4: +{pass_at_4_accuracy - pass_at_1_accuracy:.2f}%")
print()
print("=" * 80)

# ==============================================================================
# SAMPLE OUTPUTS
# ==============================================================================
print("\n" + "=" * 80)
print("📝 SAMPLE OUTPUTS (First 3 problems)")
print("=" * 80)

for idx in range(min(3, len(results))):
    r = results[idx]
    print(f"\n{'=' * 80}")
    print(f"Problem {idx+1}:")
    print(f"{'=' * 80}")
    print(f"Q: {r['problem'][:150]}...")
    print(f"\nGround Truth: {r['ground_truth']}")
    print(f"Pass@1: {'✅ CORRECT' if r['pass@1'] else '❌ WRONG'}")
    print(f"Pass@4: {'✅ CORRECT' if r['pass@4'] else '❌ WRONG'}")
    print(f"\nPredictions:")
    for i, (pred, correct) in enumerate(zip(r['predictions'], r['correct']), 1):
        status = "✅" if correct else "❌"
        print(f"  Generation {i}: {pred:>10s} {status}")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved to: {results_dir.absolute()}")
print(f"   • evaluation_metrics.json (summary)")
print(f"   • evaluation_details.json (full results)")
print()
