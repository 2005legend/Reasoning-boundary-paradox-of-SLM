#!/usr/bin/env python3
"""
Simple script to add Cell 5 (XML Prompt Formatter) to the notebook.
We'll read the file as text, find the insertion point, and insert the new cells.
"""

import re

notebook_path = r"c:\Users\USER\sidaarth\reasoning boundry paradox of SLM\rlvr_training_pipeline.ipynb"

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define Cell 5 markdown header
cell5_markdown = '''  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cell 5: XML Prompt Formatter\\n",
    "\\n",
    "**Task 2.2**: Format prompts with Qwen2.5 instruction template including XML structure expectation.\\n",
    "\\n",
    "**Requirements:** 4.6, 34.1\\n",
    "\\n",
    "This formatter creates prompts that:\\n",
    "1. Use the Qwen2.5-Instruct chat template format\\n",
    "2. Explicitly instruct the model to output in XML format with `<reasoning>` and `<answer>\\\\boxed{...}</answer>` tags"
   ]
  },'''

# Define Cell 5 code
cell5_code = '''  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import List, Optional\\n",
    "\\n",
    "def format_qwen_xml_prompt(problem: str, include_example: bool = True) -> str:\\n",
    "    \\"\\"\\"\\n",
    "    Formats a math problem prompt with Qwen2.5 instruction template and XML structure expectation.\\n",
    "    \\n",
    "    The Qwen2.5-Instruct chat format uses:\\n",
    "    <|im_start|>system\\\\\\\\n{system_message}<|im_end|>\\\\\\\\n\\n",
    "    <|im_start|>user\\\\\\\\n{user_message}<|im_end|>\\\\\\\\n\\n",
    "    <|im_start|>assistant\\\\\\\\n\\n",
    "    \\n",
    "    We include a system message that explains the expected XML format with:\\n",
    "    - <reasoning>...</reasoning> for step-by-step reasoning\\n",
    "    - <answer>\\\\\\\\\\\\\\\\boxed{...}</answer> for the final numeric answer\\n",
    "    \\n",
    "    Requirements:\\n",
    "    - 4.6: Format prompts with instruction template matching Base_Model chat format\\n",
    "    - 34.1: Display example prompts showing expected format\\n",
    "    \\n",
    "    Args:\\n",
    "        problem: The math problem text to solve\\n",
    "        include_example: Whether to include an example in the system prompt (default True)\\n",
    "    \\n",
    "    Returns:\\n",
    "        Formatted prompt string ready for model input\\n",
    "    \\"\\"\\"\\n",
    "    \\n",
    "    # System message explaining the expected XML output format\\n",
    "    if include_example:\\n",
    "        system_message = \\"\\"\\"\\"You are a helpful math problem solver. You must structure your response in XML format with the following tags:\\\\n\\n",
    "\\\\n\\n",
    "1. <reasoning>: Show your step-by-step reasoning and calculations\\\\n\\n",
    "2. <answer>: Provide the final numeric answer enclosed in \\\\\\\\\\\\\\\\boxed{}\\\\n\\n",
    "\\\\n\\n",
    "Example format:\\\\n\\n",
    "<reasoning>\\\\n\\n",
    "Let's break down the problem step by step.\\\\n\\n",
    "First, we need to calculate...\\\\n\\n",
    "Then, we multiply...\\\\n\\n",
    "Finally, we get the answer.\\\\n\\n",
    "</reasoning>\\\\n\\n",
    "<answer>\\\\\\\\\\\\\\\\boxed{42}</answer>\\\\n\\n",
    "\\\\n\\n",
    "Always use this exact XML structure in your response.\\"\\"\\"\\n",
    "    else:\\n",
    "        system_message = \\"\\"\\"\\"You are a helpful math problem solver. Structure your response with:\\\\n\\n",
    "<reasoning>Your step-by-step solution</reasoning>\\\\n\\n",
    "<answer>\\\\\\\\\\\\\\\\boxed{final_answer}</answer>\\"\\"\\"\\n",
    "    \\n",
    "    # Format with Qwen2.5-Instruct chat template\\n",
    "    formatted_prompt = f\\"\\"\\"<|im_start|>system\\\\n\\n",
    "{system_message}<|im_end|>\\\\n\\n",
    "<|im_start|>user\\\\n\\n",
    "{problem}<|im_end|>\\\\n\\n",
    "<|im_start|>assistant\\\\n\\n",
    "\\"\\"\\"\\n",
    "    \\n",
    "    return formatted_prompt\\n",
    "\\n",
    "def batch_format_prompts(\\n",
    "    problems: List[str],\\n",
    "    include_example: bool = True\\n",
    ") -> List[str]:\\n",
    "    \\"\\"\\"\\n",
    "    Formats a batch of problems with XML template.\\n",
    "    \\n",
    "    Args:\\n",
    "        problems: List of problem texts\\n",
    "        include_example: Whether to include format example in system prompt\\n",
    "    \\n",
    "    Returns:\\n",
    "        List of formatted prompts\\n",
    "    \\"\\"\\"\\n",
    "    return [format_qwen_xml_prompt(p, include_example) for p in problems]\\n",
    "\\n",
    "def display_prompt_example(problem: str, formatted_prompt: Optional[str] = None):\\n",
    "    \\"\\"\\"\\n",
    "    Displays an example of prompt formatting for inspection.\\n",
    "    \\n",
    "    Requirement 34.1: Display example prompts showing expected format\\n",
    "    \\n",
    "    Args:\\n",
    "        problem: Original problem text\\n",
    "        formatted_prompt: Pre-formatted prompt (if None, will format the problem)\\n",
    "    \\"\\"\\"\\n",
    "    if formatted_prompt is None:\\n",
    "        formatted_prompt = format_qwen_xml_prompt(problem)\\n",
    "    \\n",
    "    print(\\"=\\" * 80)\\n",
    "    print(\\"XML PROMPT FORMAT EXAMPLE\\")\\n",
    "    print(\\"=\\" * 80)\\n",
    "    print(\\"\\\\\\\\n📝 ORIGINAL PROBLEM:\\")\\n",
    "    print(\\"-\\" * 80)\\n",
    "    print(problem)\\n",
    "    print(\\"-\\" * 80)\\n",
    "    \\n",
    "    print(\\"\\\\\\\\n🔧 FORMATTED PROMPT (Qwen2.5-Instruct + XML):\\")\\n",
    "    print(\\"-\\" * 80)\\n",
    "    print(formatted_prompt)\\n",
    "    print(\\"-\\" * 80)\\n",
    "    \\n",
    "    print(\\"\\\\\\\\n📋 EXPECTED MODEL OUTPUT FORMAT:\\")\\n",
    "    print(\\"-\\" * 80)\\n",
    "    print(\\"\\"\\"<reasoning>\\\\n\\n",
    "Step 1: [First calculation or logical step]\\\\n\\n",
    "Step 2: [Next step in the solution]\\\\n\\n",
    "...\\\\n\\n",
    "Step N: [Final calculation]\\\\n\\n",
    "</reasoning>\\\\n\\n",
    "<answer>\\\\\\\\\\\\\\\\boxed{final_numeric_answer}</answer>\\"\\"\\")\\n",
    "    print(\\"-\\" * 80)\\n",
    "    print(\\"\\\\\\\\n✓ XML tags ensure structured output for reward computation\\")\\n",
    "    print(\\"✓ <reasoning> tag captures step-by-step solution process\\")\\n",
    "    print(\\"✓ <answer> tag with \\\\\\\\\\\\\\\\boxed{} enables automatic answer extraction\\")\\n",
    "    print(\\"=\\" * 80)\\n",
    "\\n",
    "# Demo the XML prompt formatter\\n",
    "if __name__ == \\"__main__\\":\\n",
    "    # Check if gsm8k_train exists from previous cell\\n",
    "    if 'gsm8k_train' in globals() and gsm8k_train['count'] > 0:\\n",
    "        print(\\"\\\\\\\\n🎯 Demonstrating XML Prompt Formatter\\")\\n",
    "        print(\\"=\\" * 80)\\n",
    "        \\n",
    "        # Get first problem from loaded dataset\\n",
    "        sample_problem = gsm8k_train['problems'][0]\\n",
    "        \\n",
    "        # Format and display\\n",
    "        formatted = format_qwen_xml_prompt(sample_problem)\\n",
    "        display_prompt_example(sample_problem, formatted)\\n",
    "        \\n",
    "        print(\\"\\\\\\\\n✅ XML prompt formatter ready!\\")\\n",
    "        print(\\"   Use format_qwen_xml_prompt(problem) to format individual problems\\")\\n",
    "        print(\\"   Use batch_format_prompts(problems) to format multiple problems\\")\\n",
    "    else:\\n",
    "        print(\\"⚠️  Note: Run Cell 4 (GSM8K loader) first to see example with real data\\")\\n",
    "        print(\\"\\\\\\\\n📝 Demo with placeholder problem:\\")\\n",
    "        demo_problem = \\"Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\\"\\n",
    "        display_prompt_example(demo_problem)"
   ]
  },'''

# Find the insertion point - after the GSM8K cell closes
# Look for the pattern that ends Cell 4
pattern = r'(    "    gsm8k_train, gsm8k_test = load_gsm8k_dataset\(max_retries=3\)"\n   \]\n  \}\n \],)'

match = re.search(pattern, content)

if match:
    insertion_point = match.end()
    
    # Insert the new cells
    new_content = (
        content[:insertion_point] +
        '\n' + cell5_markdown +
        '\n' + cell5_code +
        '\n' + content[insertion_point:]
    )
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully inserted Cell 5 into the notebook!")
    print("   Location: After Cell 4 (GSM8K Dataset Loader)")
else:
    print("❌ Could not find insertion point")
    print("Looking for pattern:", pattern)
