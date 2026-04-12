import random
from typing import List, Dict, Optional

# 25 curated snippets for code review tasks
SNIPPETS = [
    # identify_bug (Easy)
    {
        "id": "ib_001",
        "task_type": "identify_bug",
        "code_snippet": "def get_last(items):\n    return items[len(items)]",
        "bug_type": "index_error",
        "aliases": ["index error", "off by one", "fencepost error", "out of bounds", "indexerror"],
        "partial_match_words": ["index", "range", "boundary", "length"],
        "test_cases": [
            {"input_args": [[1, 2, 3]], "expected_output": 3},
            {"input_args": [["a", "b"]], "expected_output": "b"}
        ],
        "style_keywords": ["naming", "docstring"],
        "context": "Function should return the last element of a list.",
        "difficulty": "easy"
    },
    {
        "id": "ib_002",
        "task_type": "identify_bug",
        "code_snippet": "def add_numbers(a, b):\n    return a + c",
        "bug_type": "name_error",
        "aliases": ["name error", "undefined variable", "missing variable"],
        "partial_match_words": ["variable", "definition", "scope", "name"],
        "test_cases": [
            {"input_args": [1, 2], "expected_output": 3}
        ],
        "style_keywords": ["naming", "type hints"],
        "context": "Function should return the sum of two numbers.",
        "difficulty": "easy"
    },

    # suggest_fix (Medium)
    {
        "id": "sf_001",
        "task_type": "suggest_fix",
        "code_snippet": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n)",
        "bug_type": "infinite_recursion",
        "aliases": ["recursion error", "infinite recursion", "stack overflow"],
        "partial_match_words": ["recursion", "base case", "decrement", "infinite"],
        "test_cases": [
            {"input_args": [0], "expected_output": 1},
            {"input_args": [1], "expected_output": 1},
            {"input_args": [3], "expected_output": 6},
            {"input_args": [5], "expected_output": 120}
        ],
        "style_keywords": ["docstring", "type hints", "iterative"],
        "context": "Function should calculate the factorial of n recursively.",
        "difficulty": "medium"
    },
    {
        "id": "sf_007",
        "task_type": "suggest_fix",
        "code_snippet": "def get_nested_value(data, keys):\n    curr = data\n    for k in keys:\n        curr = curr[k]\n    return curr",
        "bug_type": "key_error",
        "aliases": ["key error", "missing key", "safe get"],
        "partial_match_words": ["keyerror", "get", "none", "nested", "safe"],
        "test_cases": [
            {"input_args": [{"a": {"b": 1}}, ["a", "b"]], "expected_output": 1},
            {"input_args": [{"a": 1}, ["b"]], "expected_output": None}
        ],
        "style_keywords": ["safe get", "exception handling"],
        "context": "Function should safely retrieve nested value. Return None if key is missing.",
        "difficulty": "medium"
    },
    {
        "id": "sf_009",
        "task_type": "suggest_fix",
        "code_snippet": "import json\ndef parse_api_response(resp):\n    data = json.loads(resp)\n    return data['user']['profile']['id']",
        "bug_type": "key_error",
        "aliases": ["key error", "missing key", "nested key access"],
        "partial_match_words": ["keyerror", "get", "none", "json"],
        "test_cases": [
            {"input_args": ['{"user": {"profile": {"id": 123}}}'], "expected_output": 123},
            {"input_args": ['{"user": {}}'], "expected_output": None}
        ],
        "style_keywords": ["get()", "exception handling"],
        "context": "Function parses API JSON. Must handle missing keys safely by returning None.",
        "difficulty": "hard"
    },

    # full_review (Hard)
    {
        "id": "fr_001",
        "task_type": "full_review",
        "code_snippet": "def calculate_average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    avg = total / len(nums)\n    return avg",
        "bug_type": "zero_division",
        "aliases": ["zero division", "empty list case", "division by zero"],
        "partial_match_words": ["empty", "zero", "length", "division"],
        "test_cases": [
            {"input_args": [[1, 2, 3]], "expected_output": 2.0},
            {"input_args": [[]], "expected_output": 0.0}
        ],
        "style_keywords": ["naming", "docstring", "type hints", "mean function"],
        "context": "Function should calculate average. Must handle empty lists by returning 0.0.",
        "difficulty": "hard"
    },

    # security_audit (Hard)
    {
        "id": "sa_001",
        "task_type": "security_audit",
        "code_snippet": "def delete_file(filename):\n    import os\n    os.system(\"rm \" + filename)",
        "vulnerability_keywords": ["command injection", "shell injection", "os.system"],
        "forbidden_patterns": ["os.system", "os.popen", "subprocess.Popen(..., shell=True)"],
        "secure_patterns": ["subprocess.run", "shell=False"],
        "test_cases": [
            {"input_args": ["test.txt"], "expected_output": None}
        ],
        "context": "Secure this function against shell injection.",
        "difficulty": "hard"
    },
    {
        "id": "sa_002",
        "task_type": "security_audit",
        "code_snippet": "def search_user(username):\n    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n    db.execute(query)",
        "vulnerability_keywords": ["sql injection", "string concatenation", "parameterized query"],
        "forbidden_patterns": ["db.execute"], # Heuristic for raw string query
        "secure_patterns": ["?", "%s", "execute(query, ("], # Parameterized patterns
        "test_cases": [
            {"input_args": ["alice"], "expected_output": None}
        ],
        "context": "Secure this function against SQL injection.",
        "difficulty": "hard"
    },

    # performance_refactor (Hard)
    {
        "id": "pr_001",
        "task_type": "performance_refactor",
        "code_snippet": "def has_intersection(list1, list2):\n    for x in list1:\n        if x in list2:\n            return True\n    return False",
        "performance_keywords": ["o(n^2)", "nested loop", "set", "complexity"],
        "optimized_patterns": ["set(", "dict(", "any("],
        "test_cases": [
            {"input_args": [[1, 2, 3], [3, 4, 5]], "expected_output": True},
            {"input_args": [[1, 2], [3, 4]], "expected_output": False}
        ],
        "context": "Optimize this intersection check from O(n^2) to O(n).",
        "difficulty": "hard"
    },
    {
        "id": "pr_002",
        "task_type": "performance_refactor",
        "code_snippet": "def build_string(n):\n    s = ''\n    for i in range(n):\n        s += str(i)\n    return s",
        "performance_keywords": ["string concatenation", "immutable", ".join()", "o(n^2)"],
        "optimized_patterns": [".join(", "list comprehension"],
        "test_cases": [
            {"input_args": [3], "expected_output": "012"}
        ],
        "context": "Optimize string concatenation in a loop to avoid quadratic complexity.",
        "difficulty": "hard"
    }
]

def get_task_snippet(task_type: str) -> dict:
    filtered = [s for s in SNIPPETS if s["task_type"] == task_type]
    if not filtered:
        # Fallback if no specific task found
        return random.choice(SNIPPETS)
    return random.choice(filtered)

def get_by_id(snippet_id: str) -> Optional[dict]:
    for s in SNIPPETS:
        if s["id"] == snippet_id:
            return s
    return None
