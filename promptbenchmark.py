import time
import re
import json
import pandas as pd
from pathlib import Path
from ollama import Client

# ========== CONFIG ==========
OLLAMA_URL = "ollama url"
MODEL = "llama3.1:8b"
# evalutor can be different or the same
EVALUATOR_MODEL = "llama3.1:8b"
OUTPUT_CSV = "prompt_eval.csv"
# to also try the same prompts but repeated 
ENABLE_DOUBLE_USER_PROMPT = True

client = Client(host=OLLAMA_URL)

# ========== EVALUATOR SYSTEM PROMPT ==========
EVAL_SYSTEM_PROMPT = """You are an expert evaluator of CLIL teaching materials for Austrian HTL students.
Score the provided material on the following 5 criteria, each from 1 to 10:

1. structure     — Is the material clearly structured with title, tasks and solutions section?
2. language_fit  — Does the language match the requested CEFR level and language of instruction?
3. content       — Is the subject-matter content (code, formulas, facts) accurate and complete?
4. pedagogy      — Does the material match the requested pedagogical focus and task variety?
5. vocabulary    — Are technical terms appropriately integrated and explained?

Output ONLY a JSON object. No explanation, no text before or after:
{"structure": 8, "language_fit": 7, "content": 9, "pedagogy": 8, "vocabulary": 7, "overall": 7.8}

The "overall" field must be your holistic score (not just the average), reflecting how usable this material is for a real HTL classroom.
"""

def evaluate_response(response: str, case: dict) -> dict:
    """Score a generated material using the evaluator model."""
    judge_prompt = f"""
Material type: {case['type']}
Topic: {case['topic']}
Subject: {case['subject']}
Requested language: {case['language']}
Requested CEFR level: {case['lang_level']}
Requested vocabulary emphasis: {case['vocab_emphasis']}%
Requested pedagogical focus: {case['pedagogy']}

--- GENERATED MATERIAL (first 2000 chars) ---
{response[:2000]}
--- END ---

Score this material now. Output ONLY JSON.
""".strip()

    try:
        resp = client.chat(model=EVALUATOR_MODEL, messages=[
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt}
        ])
        raw = resp["message"]["content"].strip()

        # Robust JSON extraction
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {
                "score_structure":    float(scores.get("structure", 0)),
                "score_language_fit": float(scores.get("language_fit", 0)),
                "score_content":      float(scores.get("content", 0)),
                "score_pedagogy":     float(scores.get("pedagogy", 0)),
                "score_vocabulary":   float(scores.get("vocabulary", 0)),
                "score_overall":      float(scores.get("overall", 0)),
                "eval_raw":           raw,
                "eval_error":         None,
            }
        else:
            return _empty_scores(f"JSON parse failed: {raw[:100]}")
    except Exception as e:
        return _empty_scores(str(e))

def _empty_scores(error_msg: str) -> dict:
    return {
        "score_structure": 0, "score_language_fit": 0, "score_content": 0,
        "score_pedagogy": 0, "score_vocabulary": 0, "score_overall": 0,
        "eval_raw": "", "eval_error": error_msg,
    }

# ========== SYSTEM PROMPTS ==========
SYSTEM_PROMPTS = {

    "sys_minimal": "You create student-facing teaching materials (quizzes or worksheets) for HTL students. Always include solutions at the end.",

    "sys_short": """
You are a CLIL teacher at an Austrian HTL creating teaching materials for technical subjects.
Generate quizzes or worksheets following the user's parameters exactly (language, level, topic).
Always include a clearly separated solutions section at the end.
Do not add meta-commentary — output only student-facing material and solutions.
""".strip(),

    "sys_medium": """
You are a meticulous and proficient CLIL (Content and Language Integrated Learning) teacher 
specializing in technical subjects at Austrian HTL and AHS schools.

Your role is to generate two types of teaching materials:
1. Quizzes — for assessment of knowledge and vocabulary.
2. Worksheets / Assignments — for guided student practice and independent work.

For every material you generate, strictly follow these rules:
- Respect all user-defined parameters: language, CEFR level, subject, topic, pedagogical focus, vocabulary emphasis.
- Use precise, subject-correct technical terminology at the appropriate level.
- Structure your output with: Title → Introduction → Tasks (numbered) → Solutions.
- Write only student-facing content. Do not include teacher notes or meta-commentary.
- The entire output must be written in the user-specified language of instruction.
""".strip(),

    "sys_detailed": """
You are an expert CLIL (Content and Language Integrated Learning) teacher and material designer, 
specialized in technical subjects at Austrian HTL (Höhere Technische Lehranstalt) and AHS schools. 
Your expertise covers Informatics, Programming, Electrical Engineering, Electronics, Mechanics, 
Networking, Mathematics and applied technical sciences.

You are trained to produce two types of pedagogically sound materials:
1. **Quizzes**: Assessment instruments containing 4–8 questions of varying types 
   (multiple-choice, short answer, true/false, matching, code completion).
2. **Worksheets / Assignments**: Structured independent work materials containing 4–6 scaffolded tasks, 
   designed to build progressively in difficulty.

Mandatory output rules:
- **Language**: Write the entire output in the user-specified language of instruction (German or English).
- **CEFR Level**: Calibrate vocabulary complexity, sentence structure and technical depth to the specified CEFR level.
- **Technical vocabulary**: Integrate subject-specific terminology at the requested emphasis percentage. 
  For emphasis ≥ 40%, bold key terms and add brief contextual explanations where appropriate.
- **Pedagogical focus**: Align all tasks with the stated goal (e.g. comprehension, application, problem-solving, exam-prep).
- **Format**: Use clear headings, numbered tasks, appropriate formatting (bold for key terms, code blocks for code).
- **Solutions**: Always end with a clearly separated section titled "Solutions" / "Lösungen" 
  with correct answers, sample solutions or annotated example code.
- **Accuracy**: Ensure all subject-matter content (formulas, code, circuit descriptions) is factually correct.
- Write only student-facing output. Never include teacher instructions or meta-commentary.
""".strip(),

    "sys_role_heavy": """
You will be acting as a professional CLIL teaching assistant trained to create 
high-quality classroom materials for Austrian HTL students in technical subjects. 

Your training covers:
- Informatics and Software Engineering (Python, Java, C, SQL, OOP, Algorithms, Networks)
- Electrical Engineering and Electronics (Ohm's Law, circuits, digital logic)
- Mechanical Engineering (statics, kinematics, thermodynamics)
- Mathematics (algebra, calculus, trigonometry, statistics, linear algebra)

Follow these steps for every request:
1. Read all user parameters carefully (language, CEFR level, subject, topic, vocabulary emphasis, pedagogical focus).
2. Plan the structure before writing: choose task types appropriate for the material type (quiz or worksheet).
3. Write the material in the correct language at the calibrated CEFR level.
4. Bold key technical terms and integrate them at the requested vocabulary emphasis level.
5. Ensure all technical content (code, formulas) is factually correct.
6. Add a clearly separated solutions section with detailed correct answers.
7. Output student-facing content only — no teacher notes, no meta-commentary.

The output language must always match the user-specified language of instruction.
""".strip(),
}

# ========== USER PROMPT TEMPLATES ==========

def build_user_prompt_simple(case: dict) -> str:
    return f"""
Create a {case['type']} on "{case['topic']}" for subject "{case['subject']}" 
in {case['language']} at CEFR level {case['lang_level']}.
- Vocabulary emphasis: {case['vocab_emphasis']}% technical terms
- Pedagogical focus: {case['pedagogy']}
{("- Additional notes: " + case['extra']) if case.get('extra') else ""}
Include solutions at the end.
""".strip()

def build_user_prompt_structured(case: dict) -> str:
    return f"""
Please create a {case['type'].upper()} for HTL students.

Parameters:
- Language proficiency level: {case['lang_level']} (CEFR)
- Language of instruction: {case['language']}
- Subject: {case['subject']}
- Topic: {case['topic']}
- Subject-specific vocabulary emphasis: approx. {case['vocab_emphasis']}%
- Pedagogical focus: {case['pedagogy']}
- Additional description: {case.get('extra', 'none')}

Requirements:
- Match the language and level precisely.
- Include numbered, clearly structured tasks.
- Provide a solutions section at the end.
- Output must be written entirely in {case['language']}.
""".strip()

def build_user_prompt_clil_detailed(case: dict) -> str:
    return f"""
Create a {case['type'].capitalize()} on the topic "{case['topic']}" for the subject "{case['subject']}" 
in {case['language']} using a CLIL (Content and Language Integrated Learning) approach 
with the following specifications:

**Target Audience & Language Parameters:**
- Target language proficiency level: {case['lang_level']} (CEFR)
- Language of instruction: {case['language']}
- Subject-specific vocabulary emphasis: Highlight and explain approximately {case['vocab_emphasis']}% of key terminology
- Pedagogical focus: {case['pedagogy']}

**Additional Content Requirements:**
{case.get('extra', 'None.')}

**Structure & Formatting Guidelines:**
- Use clear headings and logical section organization
- Employ appropriate formatting (bold for key terms, italics for emphasis)
- Ensure content is age-appropriate and culturally sensitive

**Content Development:**
- Integrate authentic subject content with language learning objectives
- Provide scaffolding appropriate for {case['lang_level']} learners
- Include examples, illustrations, or practice elements suitable for the material type
- Ensure accuracy of subject-matter information
- Support comprehension through context clues and supportive language

Generate well-structured, pedagogically sound content that effectively combines subject learning with language development.

IMPORTANT!!!
The whole output needs to be in the provided language of instruction: {case['language']}.
""".strip()

def build_user_prompt_kik4clil_style(case: dict) -> str:
    return f"""
Please create a {case['type'].lower()} exercise appropriate for CEFR {case['lang_level']} level 
for HTL students in the subject "{case['subject']}" on the topic "{case['topic']}".

Language requirements:
- All output must be written in {case['language']}.
- Vocabulary and sentence complexity must match CEFR {case['lang_level']}.
- Integrate technical vocabulary in approximately {case['vocab_emphasis']}% of the content.
- Bold all key technical terms on first use and provide a short in-text explanation where needed.

Pedagogical requirements:
- Pedagogical focus: {case['pedagogy']}
- Tasks must challenge students to actively demonstrate understanding.
- Use a variety of task types appropriate for a {case['type'].lower()}.
- Ensure clear student instructions for every task.

Output structure:
1. Title (include topic, subject and material type)
2. Short student introduction (2–3 sentences)
3. Tasks (numbered, clearly formatted)
4. Solutions / Lösungen (full correct answers or annotated sample code)

{("Additional notes: " + case['extra']) if case.get('extra') else ""}

Important: The entire output must be in {case['language']} only.
""".strip()

USER_PROMPT_BUILDERS = {
    "user_simple":        build_user_prompt_simple,
    "user_structured":    build_user_prompt_structured,
    "user_clil_detailed": build_user_prompt_clil_detailed,
    "user_kik4clil":      build_user_prompt_kik4clil_style,
}

TEST_CASES = [
    {
        "id": 1, "type": "quiz",
        "lang_level": "B2", "language": "English",
        "vocab_emphasis": 40, "pedagogy": "concept understanding and application",
        "topic": "Object-Oriented Programming (Inheritance & Polymorphism)",
        "subject": "Informatics / Programming",
        "extra": "Use Java-style examples."
    }
]

def run_generation(system_prompt: str, user_prompt: str, double_user: bool = False) -> dict:
    messages = [{"role": "system", "content": system_prompt}]
    final_user = (user_prompt + "" + user_prompt) if double_user else user_prompt
    messages.append({"role": "user", "content": final_user})
    start = time.time()
    try:
        resp = client.chat(model=MODEL, messages=messages)
        return {
            "response": resp["message"]["content"],
            "duration": round(time.time() - start, 2),
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
            "total_tokens": resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0),
            "gen_error": None,
        }
    except Exception as e:
        return {"response": "", "duration": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "gen_error": str(e)}

def main():
    results = []
    total = len(SYSTEM_PROMPTS) * len(USER_PROMPT_BUILDERS) * len(TEST_CASES) * (2 if ENABLE_DOUBLE_USER_PROMPT else 1)
    print(f"Model: {MODEL} | Evaluator: {EVALUATOR_MODEL}")
    print(f"Total runs: {total} (each = generation + evaluation)")
    print("=" * 60)

    counter = 0
    for sys_name, sys_prompt in SYSTEM_PROMPTS.items():
        for user_name, user_builder in USER_PROMPT_BUILDERS.items():
            for case in TEST_CASES:
                user_prompt = user_builder(case)
                for double in ([False, True] if ENABLE_DOUBLE_USER_PROMPT else [False]):
                    counter += 1
                    tag = "DOUBLE" if double else "SINGLE"
                    print(f"\n[{counter}/{total}] sys={sys_name} | user={user_name} | case={case['id']} | {tag}")

                    gen = run_generation(sys_prompt, user_prompt, double_user=double)
                    if gen["gen_error"]:
                        print(f"   Gen error: {gen['gen_error']}")
                    else:
                        print(f"   Generated: {gen['duration']}s | {gen['total_tokens']} tokens | {len(gen['response'])} chars")

                    scores = evaluate_response(gen["response"], case)
                    if scores["eval_error"]:
                        print(f"    Eval error: {scores['eval_error']}")
                    else:
                        print(f"   Scores → overall: {scores['score_overall']} | structure: {scores['score_structure']} | content: {scores['score_content']}")

                    results.append({
                        "model": MODEL,
                        "evaluator": EVALUATOR_MODEL,
                        "system_id": sys_name,
                        "user_prompt_id": user_name,
                        "case_id": case["id"],
                        "case_type": case["type"],
                        "topic": case["topic"],
                        "subject": case["subject"],
                        "lang_level": case["lang_level"],
                        "language": case["language"],
                        "vocab_emphasis": case["vocab_emphasis"],
                        "pedagogy": case["pedagogy"],
                        "double_user": double,
                        "system_prompt_len": len(sys_prompt),
                        "user_prompt_len": len(user_prompt),
                        "response_len": len(gen["response"]),
                        "response_preview": gen["response"][:400],
                        **{k: v for k, v in gen.items() if k != "response"},
                        **scores,
                    })

                    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print(f"\nDONE → {OUTPUT_CSV}")

    df = pd.DataFrame(results)
    print("\n BEST SYSTEM PROMPT (avg overall score):")
    print(df.groupby("system_id")["score_overall"].mean().round(2).sort_values(ascending=False))
    print("\n BEST USER PROMPT STYLE (avg overall score):")
    print(df.groupby("user_prompt_id")["score_overall"].mean().round(2).sort_values(ascending=False))
    print("\n SINGLE vs DOUBLE USER PROMPT:")
    print(df.groupby("double_user")["score_overall"].mean().round(2))

if __name__ == "__main__":
    main()
