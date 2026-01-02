import json
from prompt import PROMPTS
from utils import generate, extract_between_braces, cosine_similarity, match_answer, delete_certain_id_from_json, extract_between_braces_without_new_line
from concurrent.futures import ThreadPoolExecutor, as_completed
from mcts import search_for_each_distractor
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run student modeling experiments')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', 
                        help='Model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--method', type=str, default='mcts',
                        help='Method to use for distractor search (default: mcts)')
    parser.add_argument('--subject', type=str, default='eedi',
                        help='Subject dataset to use (default: eedi)')                   
    return parser.parse_args()

args = parse_args()
subject = args.subject
model = args.model
method = args.method

def generate_concept_for_each_problem(model, problem, extract_concepts_dir):
    prompt = PROMPTS["generate_concept"].format(question=problem["stem"])
    request = {
        "id": problem["problem_id"],
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "output_file": f"{extract_concepts_dir}/concept.json"
    }
    while True:
        try:
            output = generate(request)["choices"][0]["message"]["content"]
            output = extract_between_braces_without_new_line(output, chars=["[", "]"])
            if output == None:
                print(problem["problem_id"])
                print(f"{extract_concepts_dir}/concept.json")
                raise Exception("output is None")
            output = output.strip()
            this_concepts = [c.strip().upper() for c in json.loads(output)]
            break
        except Exception as e:
            print(e)
            delete_certain_id_from_json(
                file_path=extract_concepts_file,
                id=problem["problem_id"]
            )
    return problem["problem_id"], this_concepts

def generate_all_concepts_and_embeddings(model, problems, extract_concepts_dir):
    concepts, embeddings = {}, {}
    futures = []
    with ThreadPoolExecutor(max_workers=1000) as executor:
        for problem in problems:
            futures.append(executor.submit(lambda p: generate_concept_for_each_problem(*p), [model, problem, extract_concepts_dir]))
        for job in as_completed(futures):
            problem_id, this_concepts = job.result(timeout=None)
            concepts[problem_id] = this_concepts
            for c in this_concepts:
                embeddings[c] = None

    with ThreadPoolExecutor(max_workers=1000) as executor:
        for c in embeddings.keys():
            this_request = {
                "id": c,
                "model": "text-embedding-ada-002",
                "input": c,
                "output_file": f"{extract_concepts_dir}/embedding.json"
            }
            futures.append(executor.submit(lambda p: generate(*p), [this_request]))
        for job in as_completed(futures):
            job.result(timeout=None)
    
    with open(f"{extract_concepts_dir}/embedding.json") as f:
        for line in f.readlines():
            line = json.loads(line)
            if "data" not in line["output"]:
                print(line["id"])
                print(f"{extract_concepts_dir}/embedding.json")
                raise Exception("data not in output")
            embeddings[line["id"]] = line["output"]["data"][0]["embedding"]
    return concepts, embeddings

def mcts_search_for_all_distractors(model, problems, file_dir, remove_second_score=False, remove_terminal=False):
    futures = []
    reasoning_paths = {}
    with ThreadPoolExecutor(max_workers=100) as executor:
        for problem in problems:
            if "ceval" in problem["problem_id"] or "mmlu" in problem["problem_id"]:
                continue
            for i in range(len(problem["choices"])):
                if i != problem["answer"]:
                    futures.append(executor.submit(lambda p: search_for_each_distractor(*p), [model, problem, i, file_dir, search_num, remove_second_score, remove_terminal]))
        for job in as_completed(futures):
            line = job.result(timeout=None)
            reasoning_paths[line["id"]] = line
    return reasoning_paths


def summarize_error_patterns(model, student_dir, problem_map, records, reasoning_paths, concepts):
    messages = []
    for record in records:
        student_id = record["user_id"]
        os.makedirs(f"{student_dir}/{student_id}", exist_ok=True)
        this_student_cognitive_map = {}

        for j in record["train_records"]:
            this_concepts = concepts[j["problem_id"]]
            if method == "mastery":
                tmp = "Correct" if j["status"] == 1 else "Incorrect"
            else:
                if j["status"] == 0:
                    this_reasoning_path = reasoning_paths[f"{j['problem_id']}-{j['answer']}"]["reasoning_path"]
                    is_correct = False
                else:
                    this_reasoning_path = None
                    is_correct = True
                
                tmp = {
                    "stem": problem_map[j["problem_id"]]["stem"],
                    "is_correct": is_correct,
                    "reasoning_path": this_reasoning_path,
                    "answer": problem_map[j["problem_id"]]["choices"][j["answer"]]
                }
            for c in this_concepts:
                if c not in this_student_cognitive_map:
                    this_student_cognitive_map[c] = [tmp]
                else:
                    this_student_cognitive_map[c].append(tmp)
        
        with open(f"{student_dir}/{student_id}/cognitive_map_origin.json", "w") as f:
            json.dump(this_student_cognitive_map, f, ensure_ascii=False, indent=4)
        
        for key, value in this_student_cognitive_map.items():
            answer_records = ""
            if method == "mastery":
                prompt = PROMPTS["summarize_mastery"].format(kc_name=key, correctness=json.dumps(value))
            else:
                for i, v in enumerate(value):
                    answer_records += f"Record {i+1}:\n"
                    answer_records += f"Question: {v['stem']}\n"
                    if v["is_correct"]:
                        answer_records += "Correctly solved\n"
                    else:
                        answer_records += "Incorrectly solved\n"
                        answer_records += f"Reasoning: {v['reasoning_path']}\n"
                    answer_records += f"Answer: {v['answer']}\n"
                    answer_records += "\n"
                prompt = PROMPTS["summarize_error_patterns"].format(answer_records=answer_records)
            messages.append({
                "id": key,
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "output_file": f"{student_dir}/{student_id}/summarize_error_patterns.json"
            })
    
    futures = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        for r in messages:
            futures.append(executor.submit(lambda p: generate(*p), [r]))
        for job in as_completed(futures):
            job.result(timeout=None)
    
    error_patterns = {}
    for record in records:
        student_id = record["user_id"]
        error_patterns[student_id] = {}
        with open(f"{student_dir}/{student_id}/summarize_error_patterns.json", "r") as f:
            for line in f.readlines():
                line = json.loads(line)
                if "choices" not in line["output"]:
                    print(student_dir)
                    print(student_id)
                    print(line["id"])
                error_patterns[student_id][line["id"]] = line["output"]["choices"][0]["message"]["content"]
    
    return error_patterns


def extract_similar_concepts(this_concepts, concept_embeddings, this_student_error_patterns):
    extracted_concepts_and_error_patterns = []
    candidate_concepts = list(this_student_error_patterns.keys())
    candidate_concepts_embeddings = [concept_embeddings[c] for c in candidate_concepts]

    for c in this_concepts:
        c_embedding = concept_embeddings[c]
        # Calculate cosine similarity with all candidate concepts
        similarities = [cosine_similarity(c_embedding, embedding) for embedding in candidate_concepts_embeddings]
        top_1_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:1]
        top_1_concept = candidate_concepts[top_1_indices[0]]
        extracted_concepts_and_error_patterns.append({
            "concept": top_1_concept,
            "similarity": similarities[top_1_indices[0]],
            "error_pattern": this_student_error_patterns[top_1_concept]
        })
    return extracted_concepts_and_error_patterns


def predict_distractors_for_each_record(model, student_id, this_test_record, this_concepts, this_student_error_patterns, embeddings, problem_map, student_dir, train=None):
    this_problem = problem_map[this_test_record["problem_id"]]

    extracted_concepts_and_error_patterns = extract_similar_concepts(
        this_concepts=this_concepts,
        concept_embeddings=embeddings,
        this_student_error_patterns=this_student_error_patterns
    )
    error_pattern = "\n".join([f"{i['concept']}:\n{i['error_pattern']}" for i in extracted_concepts_and_error_patterns])

    base_prompt = PROMPTS["predict_distractors"]
    prompt = base_prompt + "\n\nMastery and error pattern:\n{mastery_and_error_pattern}\n\nQuestion:\n{question}\n\nCorrect answer:\n{correct_answer}\n\nOutput:".format(question=this_problem["stem"], mastery_and_error_pattern=error_pattern, correct_answer=this_problem["choices"][this_problem["answer"]])

    mid_file = f"{student_dir}/{student_id}/predict_distractors.json"
    os.makedirs(f"{student_dir}/{student_id}", exist_ok=True)
    request = {
        "id": this_test_record["problem_id"],
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "output_file": mid_file
    }

    def judge_format_correct(string):
        try:
            string = string["choices"][0]["message"]["content"]
            string = json.loads(extract_between_braces(string, chars=["{", "}"]))
            answer = string["predicted_wrong_answer"]
            reasoning = string["simulated_reasoning"]
            return True
        except Exception as e:
            print(e)
            print(string)
            return False

    output = generate(request, judge_format_correct)
    output = output["choices"][0]["message"]["content"]
    output = json.loads(extract_between_braces(output, chars=["{", "}"]))
    answer = output["predicted_wrong_answer"]
    reasoning = output["simulated_reasoning"]
    return str(answer), this_problem["choices"][this_test_record["answer"]], student_id, this_test_record["problem_id"], reasoning


def predict_distractors(model, records, concepts, embeddings, error_patterns, problem_map, student_dir):
    futures = []
    accs, reasonings, answers = {}, {}, {}
    with ThreadPoolExecutor(max_workers=100) as executor:
        for record in records:
            student_id = record["user_id"]
            train = record["train_records"]
            for j in record["test_records"]:
                if error_patterns is not None:
                    this_student_error_patterns = error_patterns[student_id]
                    this_concepts = concepts[j["problem_id"]]
                else:
                    this_concepts = None
                    this_student_error_patterns = None
                futures.append(executor.submit(lambda p: predict_distractors_for_each_record(*p), [model, student_id, j, this_concepts, this_student_error_patterns, embeddings, problem_map, student_dir, train]))
        for job in as_completed(futures):
            pred, gold, student_id, problem_id, reasoning = job.result(timeout=None)
            acc = match_answer(gold, pred)
            if student_id not in accs:
                accs[student_id] = {problem_id: acc}
            else:
                accs[student_id][problem_id] = acc
            reasonings[student_id+"-"+problem_id] = reasoning
            answers[student_id+"-"+problem_id] = pred
    return accs, reasonings, answers

def calculate_accuracy(accs):
    acc = []
    for student_id, acs in accs.items():
        for k, v in acs.items():
            acc.append(v)
    return round(sum(acc)/len(acc)*100, 1)


def main(model, records, problems, problem_map, method):
    print("Generating concepts and embeddings")
    extract_concepts_dir = f"result_{subject}/{model}/extract_concepts"
    os.makedirs(extract_concepts_dir, exist_ok=True)
    concepts, embeddings = generate_all_concepts_and_embeddings(
        model=model,
        problems=problems,
        extract_concepts_dir=extract_concepts_dir
    )
    print("Done")

    file_dir, student_dir = f"result_{subject}/{model}/{method}", f"result_{subject}/{model}/{args.method}_student"
    os.makedirs(file_dir, exist_ok=True)
    os.makedirs(student_dir, exist_ok=True)

    print("Searching for all distractors")
    reasoning_paths = mcts_search_for_all_distractors(
        model=model,
        problems=problems,
        file_dir=file_dir,
    )
    print("Done")
    
    print("Summarizing error patterns")
    error_patterns = summarize_error_patterns(
        model=model,
        student_dir=student_dir,
        problem_map=problem_map,
        records=records,
        reasoning_paths=reasoning_paths,
        concepts=concepts
    )
    print("Done")
    
    print("Predicting distractors")
    accs, reasonings, answers = predict_distractors(
        model=model,
        records=records,
        concepts=concepts,
        embeddings=embeddings,
        error_patterns=error_patterns,
        problem_map=problem_map,
        student_dir=student_dir
    )
    print("Done")
    return accs, reasonings, answers


if __name__ == "__main__":
    with open(f"{subject}/student_records.json", "r") as f:
        records = json.load(f)

    with open(f"{subject}/problems_en_w_answer.json", "r") as f:
        problems = json.load(f)
    
    problem_map = {problem["problem_id"]: problem for problem in problems}

    acc, reasoning, answers = main(model, records, problems, problem_map, method)
    print(calculate_accuracy(acc))