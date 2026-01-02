import json
import os
from retry import retry
from openai import OpenAI
import re
from math_verify import parse, verify
from math_verify.utils import timeout
from math_verify.errors import TimeoutException

@retry()
def generate(data, judge_format_correct=None):
    if os.path.isfile(data["output_file"]):
        with open(data["output_file"]) as f:
            for i in f.readlines():
                i = json.loads(i)
                if i['id'] == data["id"]:
                    return i["output"]
    
    if 'llama' in data['model']:
        client = OpenAI(api_key="your key", base_url="your url", timeout=60)
    elif 'claude' in data['model']:
        client = OpenAI(api_key="your key", base_url='your url', timeout=60)
    elif 'deepseek' in data['model']:
        client = OpenAI(api_key="your key", base_url='your url' ,timeout=60)
    else:
        client = OpenAI(api_key='your key', base_url='your url' ,timeout=60)

    while True:
        try:
            if "embedding" in data["model"]:
                stream = client.embeddings.create(
                    model=data["model"],
                    input=data["input"],
                    encoding_format="float"
                ).to_dict()
            elif "generation_config" in data:
                if 'claude' in data['model']:
                    n = data['generation_config']["n"]
                    stream = client.chat.completions.create(
                        model=data["model"],
                        messages=data["messages"],
                        **data["generation_config"]
                    ).to_dict()
                    for _ in range(n-1):
                        this_stream = client.chat.completions.create(
                            model=data["model"],
                            messages=data["messages"],
                            **data["generation_config"]
                        ).to_dict()
                        stream["choices"].append(this_stream["choices"][0])
                else:
                    stream = client.chat.completions.create(
                        model=data["model"],
                        messages=data["messages"],
                        **data["generation_config"]
                    ).to_dict()
            else:
                stream = client.chat.completions.create(
                    model=data["model"],
                    messages=data["messages"],
                ).to_dict()
            
            if stream == {}:
                raise Exception("stream is empty")
            if judge_format_correct is not None:
                if not judge_format_correct(stream):
                    raise Exception("format incorrect")
            output_data = {
                "id": data["id"],
                "output": stream,
                "input": data
            }
            write_jsonl(file_path=data["output_file"], single_data=output_data)
                
            return output_data["output"]
        except Exception as e:
            print(e)
            print(data["id"])
            print(data["output_file"])
            # raise e

def read_json(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path) as f:
        return json.load(f)

def read_jsonl(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path) as f:
        result = [json.loads(line) for line in f.readlines()]
    result = sorted(result, key=lambda x: x['id'])
    return result

def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_jsonl(file_path, single_data):
    try:
        with open(file_path, 'a') as f:
            f.write(json.dumps(single_data, ensure_ascii=False)+'\n')
    except Exception as e:
        print(e)
        print(single_data)
        raise e

def delete_certain_id_from_json(file_path, id):
    try:
        tmp = []
        with open(file_path) as f:
            for i in f.readlines():
                i = json.loads(i)
                if i['id'] != id:
                    tmp.append(i)
        with open(file_path, 'w') as f:
            for i in tmp:
                line = json.dumps(i, ensure_ascii=False)
                f.write(line+'\n')
    except Exception as e:
        print(e)
        print(file_path)
        print(id)
        raise e

def extract_answer_text(input_string):
    """
    Extract text between <answer> and </answer> tags in a string.
    
    Args:
        input_string (str): The input string containing answer tags
        
    Returns:
        str: The extracted text between the tags, or None if no tags found
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, input_string, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None

# @timeout(10)
def verification_method(gold, answer):
    parsed_gold = parse(gold, parsing_timeout=None)
    parsed_answer = parse(answer, parsing_timeout=None)
    return verify(parsed_gold, parsed_answer, timeout_seconds=None)

@retry()
def match_answer(gold, answer):
    # If either gold or answer is None, treat as not matching
    if gold is None or answer is None:
        return False

    if gold.strip().lower() == answer.strip().lower():
        return True
    
    return verification_method(gold.strip(), answer.strip())

def extract_entity_pattern(input_string):
    """
    Extract all patterns in the format (a|b|c) from a string.
    
    Args:
        input_string (str): The input string containing entity patterns
        
    Returns:
        list: List of tuples, where each tuple contains the three parts (a, b, c)
    """
    pattern = r'\(([^|]+)\|([^|]+)\|([^)]+)\)'
    matches = re.findall(pattern, input_string)
    
    return matches

def extract_between_braces(s, chars = ['{', '}']):
    """
    Extract the substring between the first '{' and the last '}' in a string.
    Returns the substring, or None if braces are not found in the correct order.
    """

    start = -1
    for i in range(len(s)):
        if s[i] == chars[0]:
            if i + 1 < len(s) and s[i + 1] == '\n':
                start = i
                break
    
    if start == -1:
        return None
    
    end = -1
    for i in range(len(s) - 1, -1, -1):
        if s[i] == chars[1]:
            if i == len(s) - 1 or s[i + 1] == '\n':
                end = i
                break
    
    if end != -1 and end > start:
        return s[start:end+1].replace('\n', ' ')
    return None

def extract_between_braces_without_new_line(s, chars = ['{', '}']):
    start = s.find(chars[0])
    end = s.rfind(chars[1])
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two one-dimensional embeddings.
    
    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector
        
    Returns:
        float: Cosine similarity value between -1 and 1, where higher values indicate higher similarity
    """
    if len(embedding1) != len(embedding2):
        raise ValueError("Embeddings must have the same length")
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in embedding1) ** 0.5
    magnitude2 = sum(a * a for a in embedding2) ** 0.5
    
    # Handle zero magnitude case
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude1 * magnitude2)
    
    return cosine_sim