import random
from typing import List, Dict, Optional

# 18 curated snippets for code review tasks
SNIPPETS = [
    # identify_bug (Easy)
    {
        "id": "ib_001",
        "task_type": "identify_bug",
        "code_snippet": "def get_last(items):\n    return items[len(items)]",
        "bug_type": "index_error",
        "aliases": ["index error", "off by one", "fencepost error", "out of bounds"],
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
    {
        "id": "ib_003",
        "task_type": "identify_bug",
        "code_snippet": "def divide(a, b):\n    return a / 0",
        "bug_type": "zero_division",
        "aliases": ["zero division", "division by zero", "zerodivisionerror"],
        "partial_match_words": ["zero", "divide", "division", "math"],
        "test_cases": [
            {"input_args": [10, 2], "expected_output": 5.0}
        ],
        "style_keywords": ["error handling"],
        "context": "Function should divide a by b.",
        "difficulty": "easy"
    },
    {
        "id": "ib_004",
        "task_type": "identify_bug",
        "code_snippet": "def check_even(n):\n    if n % 2 = 0:\n        return True",
        "bug_type": "syntax_error",
        "aliases": ["syntax error", "invalid syntax", "assignment instead of comparison"],
        "partial_match_words": ["equals", "assignment", "syntax", "if"],
        "test_cases": [
            {"input_args": [4], "expected_output": True}
        ],
        "style_keywords": ["spacing"],
        "context": "Function should return True if n is even.",
        "difficulty": "easy"
    },
    {
        "id": "ib_005",
        "task_type": "identify_bug",
        "code_snippet": "def greet(name):\n    return \"Hello \" + name + 5",
        "bug_type": "type_error",
        "aliases": ["type error", "incompatible types", "string and int concatenation"],
        "partial_match_words": ["type", "string", "integer", "conversion"],
        "test_cases": [
            {"input_args": ["Alice"], "expected_output": "Hello Alice"}
        ],
        "style_keywords": ["f-string", "type hints"],
        "context": "Function should greet the user by name.",
        "difficulty": "easy"
    },
    {
        "id": "ib_006",
        "task_type": "identify_bug",
        "code_snippet": "def find_key(d, key):\n    return d[key_name]",
        "bug_type": "name_error",
        "aliases": ["name error", "undefined variable", "typo"],
        "partial_match_words": ["variable", "key", "name", "typo"],
        "test_cases": [
            {"input_args": [{"a": 1}, "a"], "expected_output": 1}
        ],
        "style_keywords": ["naming", "typing"],
        "context": "Function should return the value for a given key in a dictionary.",
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
        "id": "sf_002",
        "task_type": "suggest_fix",
        "code_snippet": "def find_max(numbers):\n    max_val = 0\n    for n in numbers:\n        if n > max_val:\n            max_val = n\n    return max_val",
        "bug_type": "logic_error",
        "aliases": ["wrong initialization", "logic error", "all negative numbers"],
        "partial_match_words": ["negative", "initialization", "zero", "max"],
        "test_cases": [
            {"input_args": [[1, 5, 2]], "expected_output": 5},
            {"input_args": [[-1, -5, -2]], "expected_output": -1},
            {"input_args": [[-10]], "expected_output": -10}
        ],
        "style_keywords": ["built-in function", "naming"],
        "context": "Function should find the maximum value in a list of numbers, including negative ones.",
        "difficulty": "medium"
    },
    {
        "id": "sf_003",
        "task_type": "suggest_fix",
        "code_snippet": "def remove_duplicates(items):\n    unique = []\n    for i in items:\n        if i not in unique:\n            unique.append(i)\n    return unique",
        "bug_type": "efficiency",
        "aliases": ["slow", "o(n^2)", "use a set"],
        "partial_match_words": ["set", "efficiency", "performance", "complexity"],
        "test_cases": [
            {"input_args": [[1, 2, 2, 3]], "expected_output": [1, 2, 3]},
            {"input_args": [["a", "a", "b"]], "expected_output": ["a", "b"]}
        ],
        "style_keywords": ["set literal", "performance"],
        "context": "Function should remove duplicates from a list while maintaining order, but needs to be efficient.",
        "difficulty": "medium"
    },
    {
        "id": "sf_004",
        "task_type": "suggest_fix",
        "code_snippet": "def parse_int(s):\n    try:\n        return int(s)\n    except:\n        return None",
        "bug_type": "broad_exception",
        "aliases": ["bare except", "broad exception", "catch specific error"],
        "partial_match_words": ["valueerror", "exception", "specific", "broad"],
        "test_cases": [
            {"input_args": ["123"], "expected_output": 123},
            {"input_args": ["abc"], "expected_output": None}
        ],
        "style_keywords": ["specific exception", "logging"],
        "context": "Function should safely parse a string to an integer, catching only ValueErrors.",
        "difficulty": "medium"
    },
    {
        "id": "sf_005",
        "task_type": "suggest_fix",
        "code_snippet": "def merge_dicts(d1, d2):\n    return d1.update(d2)",
        "bug_type": "inplace_modification",
        "aliases": ["returns none", "inplace update", "mutates input"],
        "partial_match_words": ["none", "return", "update", "merge", "copy"],
        "test_cases": [
            {"input_args": [{"a": 1}, {"b": 2}], "expected_output": {"a": 1, "b": 2}}
        ],
        "style_keywords": ["dictionary unpacking", "merging operator"],
        "context": "Function should merge two dictionaries and return the result as a NEW dictionary.",
        "difficulty": "medium"
    },
    {
        "id": "sf_006",
        "task_type": "suggest_fix",
        "code_snippet": "def fibonacci(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    return fibonacci(n-1) + fibonacci(n-1)",
        "bug_type": "logic_error",
        "aliases": ["wrong formula", "fibonacci logic", "repeated term"],
        "partial_match_words": ["n-2", "formula", "logic", "fibonacci"],
        "test_cases": [
            {"input_args": [0], "expected_output": 0},
            {"input_args": [1], "expected_output": 1},
            {"input_args": [2], "expected_output": 1},
            {"input_args": [5], "expected_output": 5}
        ],
        "style_keywords": ["memoization", "iteration"],
        "context": "Function should calculate the n-th Fibonacci number.",
        "difficulty": "medium"
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
    {
        "id": "fr_002",
        "task_type": "full_review",
        "code_snippet": "def get_file_content(path):\n    f = open(path, 'r')\n    data = f.read()\n    return data",
        "bug_type": "resource_leak",
        "aliases": ["file leak", "missing close", "with statement", "resource leak"],
        "partial_match_words": ["close", "handle", "leak", "context manager"],
        "test_cases": [
            {"input_args": ["test.txt"], "expected_output": "test data"}
        ],
        "style_keywords": ["context manager", "error handling", "pathlib"],
        "context": "Function should read file content. Must close the file properly.",
        "difficulty": "hard"
    },
    {
        "id": "fr_003",
        "task_type": "full_review",
        "code_snippet": "def update_list(val, my_list=[]):\n    my_list.append(val)\n    return my_list",
        "bug_type": "mutable_default_argument",
        "aliases": ["mutable default", "argument persistence", "shared list"],
        "partial_match_words": ["default", "mutable", "none", "persistent"],
        "test_cases": [
            {"input_args": [1], "expected_output": [1]},
            {"input_args": [2], "expected_output": [2]}
        ],
        "style_keywords": ["none check", "naming", "type hints"],
        "context": "Function should append value to list. Default list should not persist across calls.",
        "difficulty": "hard"
    },
    {
        "id": "fr_004",
        "task_type": "full_review",
        "code_snippet": "def square_roots(nums):\n    import math\n    roots = []\n    for n in nums:\n        roots.append(math.sqrt(n))\n    return roots",
        "bug_type": "domain_error",
        "aliases": ["negative number", "math domain error", "valueerror"],
        "partial_match_words": ["negative", "complex", "sqrt", "error"],
        "test_cases": [
            {"input_args": [[4, 9]], "expected_output": [2.0, 3.0]},
            {"input_args": [[-1]], "expected_output": []}
        ],
        "style_keywords": ["list comprehension", "top-level import", "error handling"],
        "context": "Function should return roots of positive numbers. Skip negative numbers.",
        "difficulty": "hard"
    },
    {
        "id": "fr_005",
        "task_type": "full_review",
        "code_snippet": "def get_api_data(url, retry=3):\n    for i in range(retry):\n        resp = requests.get(url)\n        if resp.status_code == 200:\n            return resp.json()\n    return None",
        "bug_type": "missing_timeout",
        "aliases": ["no timeout", "hanging request", "unbounded wait"],
        "partial_match_words": ["timeout", "blocking", "request", "hang"],
        "test_cases": [
            {"input_args": ["http://api.com"], "expected_output": {"data": "ok"}}
        ],
        "style_keywords": ["session", "backoff", "exception handling"],
        "context": "Function fetches data with retries. Must have a timeout to prevent hanging.",
        "difficulty": "hard"
    },
    {
        "id": "fr_006",
        "task_type": "full_review",
        "code_snippet": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True",
        "bug_type": "efficiency",
        "aliases": ["inefficient", "o(n)", "check until sqrt"],
        "partial_match_words": ["sqrt", "efficiency", "performance", "complexity"],
        "test_cases": [
            {"input_args": [2], "expected_output": True},
            {"input_args": [4], "expected_output": False},
            {"input_args": [11], "expected_output": True}
        ],
        "style_keywords": ["math.sqrt", "type hints", "docstring"],
        "context": "Function checks if n is prime. Needs to be efficient for large primes.",
        "difficulty": "hard"
    },
    # security_audit (Hard)
    {
        "id": "sa_001",
        "task_type": "security_audit",
        "code_snippet": "def delete_file(filename):\n    import os\n    os.system(\"rm \" + filename)",
        "bug_type": "command_injection",
        "aliases": ["os.system injection", "command injection", "shell injection"],
        "partial_match_words": ["os.system", "shell", "subprocess", "injection", "sanitize"],
        "test_cases": [
            {"input_args": ["test.txt"], "expected_output": None}
        ],
        "style_keywords": ["subprocess.run", "shell=False"],
        "context": "Function deletes a file using shell command. Needs to be secured against command injection.",
        "difficulty": "hard"
    },
    # performance_refactor (Hard)
    {
        "id": "pr_001",
        "task_type": "performance_refactor",
        "code_snippet": "def has_intersection(list1, list2):\n    for x in list1:\n        if x in list2:\n            return True\n    return False",
        "bug_type": "algorithmic_inefficiency",
        "aliases": ["o(n^2)", "nested loop", "inefficient search"],
        "partial_match_words": ["set", "hash", "complexity", "performance", "n^2"],
        "test_cases": [
            {"input_args": [[1, 2, 3], [3, 4, 5]], "expected_output": True},
            {"input_args": [[1, 2], [3, 4]], "expected_output": False}
        ],
        "style_keywords": ["set conversion", "any()"],
        "context": "Function checks for element intersection. Needs to be optimized for time complexity.",
        "difficulty": "hard"
    }
]

def get_task_snippet(task_type: str) -> dict:
    filtered = [s for s in SNIPPETS if s["task_type"] == task_type]
    return random.choice(filtered)

def get_by_id(snippet_id: str) -> Optional[dict]:
    for s in SNIPPETS:
        if s["id"] == snippet_id:
            return s
    return None
