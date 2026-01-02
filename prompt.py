PROMPTS = {}

PROMPTS["generate_concept"] = """You are given a question from a mathematics or science test. Your task is to identify all the knowledge concepts that are required to solve this question. Focus on the core concepts involved in understanding and solving the problem.

Please output the result as a Python list of strings.
Do not include any explanation or extra text.

Example output format:
["Fractions", "Least common multiple", "Subtraction"]

Question:
{question}

Output:
"""

PROMPTS["summarize_error_patterns"] = """You are given a set of answer records from a student related to the same knowledge concept. Each record includes the question, the student's reasoning, and whether their answer is correct.

Your task is to identify a specific, repeatable error pattern the student tends to make when dealing with this concept. The error pattern should be described in such a way that it can be reproduced on a new similar question, even without re-reading the original answers.

Avoid long explanations. Focus only on extracting a precise, structured pattern of the student's mistake, along with 1–2 examples that reflect this pattern.

If all answers are correct, then no error pattern is needed. Just output "Great mastery".

Answer records:
{answer_records}

Output:
"""

PROMPTS["predict_distractors"] = """You are given a new question, and it is known that the student answered it incorrectly.

You are also given:

1. The knowledge concept this question tests.

2. The student’s known mastery level of this concept.

3. The student’s common error pattern when dealing with this concept.

4. The correct answer to the question. (So that you can know the correct answer is not the one you predicted).

Your task is to simulate the student’s mistaken reasoning process based on their past error pattern, and predict what incorrect answer the student most likely gave.
The reasoning and answer must strictly follow the described error pattern.

Output format (Python dict):
{
  "simulated_reasoning": "<step-by-step reasoning that reflects the error pattern>",
  "predicted_wrong_answer": "<final wrong answer, only the final numerical value or mathematical expression>"
}"""


PROMPTS["evaluate_reasoning_process"] = """You are an expert math teacher evaluating whether a model-generated reasoning process could plausibly reflect a real student's thinking. You will be given:

1. A math question

2. A student’s final answer

3. A reasoning process that explains how the student may have arrived at the answer

Note: The reasoning may contain errors — this is expected. Your task is not to judge whether the final answer is correct, but rather to evaluate the realism and coherence of the reasoning process.

You should score the reasoning on two dimensions, each from 1 to 5 (higher is better):

1. Plausibility (Realism of the Mistake)
Does the reasoning reflect a common, cognitively plausible mistake that a real student might make?

  5 = Very realistic; reflects a well-known student misconception

  4 = Realistic and understandable

  3 = Somewhat plausible, though slightly uncommon

  2 = Weakly plausible; seems unnatural

  1 = Implausible or clearly artificial mistake

2. Causal Coherence (Logical Consistency)
Does the reasoning process follow a clear, step-by-step logic that leads to the final answer?

  5 = Fully coherent, each step follows naturally

  4 = Mostly coherent, small gaps

  3 = Reasonably coherent, but with noticeable jumps

  2 = Weakly coherent, hard to follow

  1 = Incoherent or unrelated steps

Output Format (JSON):
{
  "plausibility": <1–5>,
  "causal_coherence": <1–5>,
  "explanation": "<brief explanation of both scores>"
}
"""