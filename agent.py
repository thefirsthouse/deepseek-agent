import os
import json
import subprocess
from pathlib import Path
from typing import Any, Callable
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# --- Load .env relative to this file ---
load_dotenv(dotenv_path=str(Path(__file__).with_name(".env")))

# --- Normalize socks:// -> socks5:// for httpx2 ---
for _k in ("ALL_PROXY", "all_proxy"):
    _v = os.environ.get(_k)
    if _v and _v.startswith("socks://"):
        os.environ[_k] = "socks5://" + _v[len("socks://"):]

# --- Provider fallback chain ---
def _valid(key):
    if not key:
        return False
    if "ВАШ" in key or "your-key" in key:
        return False
    return True

PROVIDERS = [
    {
        "name": "gemini",
        "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    },
    {
        "name": "groq",
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    },
    {
        "name": "openrouter",
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "model": os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free"),
    },
    {
        "name": "deepseek",
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
]
PROVIDERS = [p for p in PROVIDERS if _valid(p["api_key"])]
if not PROVIDERS:
    raise RuntimeError("Ни один провайдер не настроен. Проверьте .env")

ENC = tiktoken.get_encoding("cl100k_base")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "60000"))


# --- Tools ---
def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: записано {len(content)} символов в {path}"
    except Exception as e:
        return f"ERROR: {e}"


def run_command(command: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return f"exit_code={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    except subprocess.TimeoutExpired:
        return f"ERROR: таймаут {timeout}с"
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"read_file": read_file, "write_file": write_file, "run_command": run_command}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Прочитать файл по указанному пути",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Создать или перезаписать файл",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Выполнить shell-команду",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}, "required": ["command"]}}},
]


# --- Context management ---
def count_tokens(messages):
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            total += len(ENC.encode(content))
        total += 4
    return total


def trim_context(messages, max_tokens=MAX_CONTEXT_TOKENS):
    if count_tokens(messages) <= max_tokens:
        return messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    while rest and count_tokens(system_msgs + rest) > max_tokens:
        rest.pop(0)
    return system_msgs + rest


# --- Fallback across providers ---
def call_with_fallback(messages):
    last_error = None
    for provider in PROVIDERS:
        client = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])
        try:
            print(f"[provider] пробуем {provider['name']} ({provider['model']})")
            return client.chat.completions.create(
                model=provider["model"],
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            print(f"[fallback] {provider['name']} не сработал: {type(e).__name__}")
            last_error = e
            continue
    if last_error:
        print(f"[fallback] все провайдеры исчерпаны: {last_error}")
    return None


def execute_tool(name, arguments):
    fn = TOOLS.get(name)
    if not fn:
        return f"ERROR: неизвестный инструмент {name}"
    try:
        return fn(**arguments)
    except Exception as e:
        return f"ERROR: {e}"


# --- Agent loop ---
def run_agent(user_task, max_iterations=10):
    messages = [
        {"role": "system", "content": "Ты автономный агент с инструментами для работы с файлами и shell-командами. Действуй пошагово."},
        {"role": "user", "content": user_task},
    ]
    for iteration in range(max_iterations):
        messages = trim_context(messages)
        response = call_with_fallback(messages)
        if response is None:
            return "Все провайдеры недоступны. Проверьте ключи и лимиты."
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "(пустой ответ)"
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    return f"Достигнут лимит итераций ({max_iterations})."


if __name__ == "__main__":
    import sys
    task = " ".join(sys.argv[1:]) or "Создай файл hello.py с print('Hello!') и запусти его."
    print(f"Задача: {task}\n")
    print(run_agent(task))
