from ollama import Client
import time
import pandas as pd
import numpy as np
import re
import json
import requests
from typing import Dict

# ========== SERVER CONFIG ==========
# Ändere dies zu der Ollama url
OLLAMA_URL = ""

client = Client(host=OLLAMA_URL)

# Modelle, die nicht getestet werden sollen (z.B. Embeddings)
EXCLUDED_MODELS = ['nomic-embed-text:latest', 'nomic-embed-text']

# Evaluator Modell festlegen z.B. qwen2.5-coder
EVALUATOR_MODEL = 'qwen2.5-coder:7b'

# ========== PROMPTS & SYSTEM ==========
MAIN_SYSTEM_PROMPT = """You are a CLIL teacher creating student worksheets and tests in technical English for Austrian HTL/AHS. 
Follow instructions precisely: generate exactly the requested number of items, include clear student instructions, 
and provide correct solutions/answers at the end in a separate section. Use B1/B2 level language."""

EVAL_SYSTEM_PROMPT = """You are an expert HTL/AHS teacher evaluator. Score this worksheet 1-10 on: clarity, correctness, completeness, pedagogy. 
Output ONLY JSON like: {"overall": 8.5, "clarity": 9, "correctness": 8, "completeness": 9, "pedagogy": 8}. No other text."""

PROMPTS = [
    # ========== Mathematik (4 Prompts) ==========

    # 1. Short Answer (Algebra/Gleichungen)
    "Please create a short-answer math exercise appropriate for HTL/AHS B2 level with 3 questions on solving quadratic equations. Include correct numeric or algebraic answers for each.",

    # 2. Multiple Choice (Differentialrechnung)
    "Please create a multiple-choice math worksheet for AHS/HTL B2 with 4 questions on calculus derivatives. Provide the correct choice and a brief explanation for each.",

    # 3. Sequencing (Geometrie/Trigonometrie)
    "Please create a sequencing math task for HTL B1/B2 with 2 problems on calculating the area of complex geometric shapes using trigonometry. List the step-by-step calculation to solve the problem and the final answer.",

    # 4. True/False ustify (Wahrscheinlichkeitsrechnung/Statistik)
    "Please create a true/false justify math exercise for HTL B2 with 4 statements on basic probability theory and normal distribution. Provide the correct answer and a brief mathematical justification.",

    # ========== Software & IT (3 Prompts) ==========

    # 5. Logic & Algorithms (Python - Short Answer)
    "Please create a short-answer programming exercise appropriate for HTL B2 level with 3 tasks in Python on array manipulation and for-loops. Provide correct code snippets for each.",

    # 6. Object Oriented Programming (Java - True/False Justify)
    "Please create a true/false justify programming exercise for HTL B2 with 4 statements on Object-Oriented Programming (inheritance and polymorphism) in Java. Provide the correct answer and a brief explanation.",

    # 7. Bug Fixing / Databases (SQL & Logic)
    "Create an exercise for HTL students where you provide a short SQL query containing a logical JOIN bug. Ask the students to identify the bug and provide the corrected query based on a simple 'School' database schema.",

    # ========== Klassische Technik / Hardware (3 Prompts) ==========

    # 8. Elektrotechnik (Multiple Choice)
    "Please create a multiple-choice exercise for HTL B2 level with 4 questions on basic electrical engineering (Ohm's law, Kirchhoff's circuit laws). Provide the correct choice and a brief explanation for each.",

    # 9. Elektronik / Digitaltechnik (Short Answer)
    "Please create a short-answer technical exercise for HTL B2 with 3 questions on digital logic gates (AND, OR, XOR) and boolean algebra. Include the correct logical expressions or truth table outcomes.",

    # 10. Maschinenbau / Mechanik (Word Problem / Applied Task)
    "Create an applied physics/mechanics word problem for HTL B2 students involving statics, specifically calculating the support reactions of a simply supported beam with a point load. Provide the problem text and the step-by-step mathematical solution."
]

OUTPUT_CSV = 'output.csv'


def get_available_models() -> list:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "models" in data:
                all_models = [m["name"] for m in data["models"] if "name" in m]
                # Filter excluded models
                test_models = [m for m in all_models if m not in EXCLUDED_MODELS]
                return sorted(test_models)
        print(f" Server antwortete mit Status Code: {response.status_code}")
    except Exception as e:
        print(f" Ollama models fetch failed: {e}")
    return []


def query_model(model: str, prompt: str, system: str) -> Dict:
    print(f"  → Querying {model}...")
    start = time.time()
    try:
        resp = client.chat(model=model, messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ])
        duration = time.time() - start
        print(f"  → OK ({duration:.1f}s, {resp.get('eval_count', 0)} tokens)")
        return {
            'response': resp['message']['content'],
            'duration': round(duration, 2),
            'tokens_prompt': resp.get('prompt_eval_count', 0),
            'tokens_completion': resp.get('eval_count', 0),
            'total_tokens': resp.get('eval_count', 0) + resp.get('prompt_eval_count', 0)
        }
    except Exception as e:
        print(f"  → ERROR: {e}")
        return {'response': f"Error: {str(e)}", 'duration': 0, 'tokens_prompt': 0, 'tokens_completion': 0,
                'total_tokens': 0}


def evaluate_response(response: str, prompt: str, evaluator: str) -> Dict:
    print(f"  → Evaluating with {evaluator}...")
    judge_prompt = f"Prompt: {prompt[:80]}\nResponse: {response[:1500]}\nScore JSON only."
    try:
        resp = client.chat(model=evaluator, messages=[
            {'role': 'system', 'content': EVAL_SYSTEM_PROMPT},
            {'role': 'user', 'content': judge_prompt}
        ])
        score_text = resp['message']['content'].strip()

        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', score_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            print(f"  → Parsed Score: {scores.get('overall', '?')}")
            return {
                'overall_score': float(scores.get('overall', 0)),
                'clarity': float(scores.get('clarity', 0)),
                'correctness': float(scores.get('correctness', 0)),
                'completeness': float(scores.get('completeness', 0)),
                'pedagogy': float(scores.get('pedagogy', 0)),
                'raw_eval': score_text
            }
    except Exception as e:
        print(f"  → Eval error: {e}")
    return {'overall_score': 0, 'clarity': 0, 'correctness': 0, 'completeness': 0, 'pedagogy': 0, 'raw_eval': 'Failed'}


print(f"Starte Ollama Evaluation")

MODELS = get_available_models()

if not MODELS:
    print("\nKeine Modelle gefunden.")
    exit()

print(f"\n Gefundene Test-Modelle ({len(MODELS)}): {MODELS}")
print(f" Evaluator-Modell: {EVALUATOR_MODEL}")

try:
    all_raw_models = [m["name"] for m in requests.get(f"{OLLAMA_URL}/api/tags").json().get("models", [])]
    if EVALUATOR_MODEL not in all_raw_models:
        print(f"\n WARNUNG: Evaluator Modell '{EVALUATOR_MODEL}' wurde nicht auf dem Server gefunden!")
        print("Das Skript wird vermutlich fehlschlagen. Bitte lade das Modell vorher auf den Server.")
except:
    pass

results = []
for i, model in enumerate(MODELS, 1):
    print(f"\n [{i}/{len(MODELS)}] Teste {model}...")
    model_times = []

    for j, prompt in enumerate(PROMPTS, 1):
        print(f"   Prompt {j}/20...")

        # 1. Generierung
        output = query_model(model, prompt, MAIN_SYSTEM_PROMPT)

        # 2. Evaluation
        scores = evaluate_response(output['response'], prompt, EVALUATOR_MODEL)

        model_times.append(output['duration'])

        if j <= 4:
            cat = 'Mathematik'
        elif j <= 7:
            cat = 'Software & IT'
        else:
            cat = 'Klassische Technik'

        result = {
            'model': model,
            'prompt_id': j,
            'category': cat,
            'prompt_preview': prompt[:100],
            'response_preview': output['response'][:300],
            'full_response_len': len(output['response']),
            **output, **scores
        }

        results.append(result)

        # Zwischenspeichern nach jedem Prompt
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"  ✅ Row {len(results)} saved | Score: {scores['overall_score']:.1f}")

    print(f"   {model} abgeschlossen. Durchschnittszeit: {np.mean(model_times):.1f}s")

df = pd.DataFrame(results)
summary = df.groupby('model')[['overall_score', 'duration']].mean().round(2).sort_values('overall_score',
                                                                                         ascending=False)
print(summary)
