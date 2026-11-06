import ollama
from ollama import ResponseError
import json
import os
import requests
import subprocess
import re
import time
import lassi.argostream_curl
import lassi.token_analysis

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)

# Argo Gateway username — set this via lassi.llm_config.argo_username = "your_username"
# (Argo Gateway is reachable only from within the Argonne National Laboratory network.)
argo_username = "enter-your-Argonne-domain-username"

# OpenAI-compatible backend connection details — set via run_lassi.py from config.yaml's
# llm.openai_compatible block. api_key falls back to the OPENAI_API_KEY environment
# variable at call time if left blank, so a key never has to live in a config file.
openai_base_url = "https://api.openai.com/v1"
openai_api_key = ""

"""
Local hosted LLM via OLLAMA API, GGUF quantized models

Requirement: ollama serve [from server command line]
Requirement: ollama.pull('modelname') [from Python script to pre-load selected model at the server]

ollama.list() [preview what is available to use on the server]

Example: model='codestral:latest'
"""


def ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):

    if verifyprompt:
        _tok = lassi.token_analysis.token_count(system_prompt + " " + content_prompt)
        print(f"\n=====> VERIFY THE PROMPTING <===== [tokens: {_tok:,}]\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
    
    response_text = ""

    # Ollama serve must already be running on a port on this server 
    response = ollama.chat(
        model=model, 
        messages=[
            {"role": "system", "content": system_prompt},
            {'role': 'user', 'content': content_prompt}
        ],
        stream=False,
        options=dict(
            temperature=temp,
            top_p=0.9,
            num_ctx=num_ctx
        )
    )

    response_text = response['message']['content']

    return response_text


def ollama_llm(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):

    if verifyprompt:
        _tok = lassi.token_analysis.token_count(system_prompt + " " + content_prompt)
        print(f"\n=====> VERIFY THE PROMPTING <===== [tokens: {_tok:,}]\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
            
    response_text = ""

    # Ollama serve must already be running on a port on this server 
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt}
        ],
        stream=True,
        options=dict(
            temperature=temp,
            top_p=0.9
            # num_ctx=num_ctx  # including this parameter seems to slow down the streaming response
        ),
        keep_alive=0
    )

    # Process each chunk in the stream
    try:
        for chunk in stream:
            chunk_content = chunk['message']['content']
            print(chunk_content, end='', flush=True)
            response_text += chunk_content
        
    except ResponseError as e:
        if "unexpected EOF" in str(e):
            # Log the event, check if the output seems complete, and continue
            print("Stream ended unexpectedly but continuing with received output.")
        else:
            raise
    
    # Clear the streaming display after completion
    # clear_output(wait=True)

    return response_text


def unload_ollama_model(model):
    print("\nUnloading Ollama server...")
    data_curl = {
        "model": model,
        "keep_alive": 0
    }
    data_curl_json = json.dumps(data_curl)
    curl_command_unload_LLM = [
        'curl',
        'http://localhost:11434/api/generate',
        '-d',
        data_curl_json
    ]
    result_unload_LLM = subprocess.run(curl_command_unload_LLM, capture_output=True, text=True)
    print("\nUnload LLM:\n", result_unload_LLM.stdout)
    print(result_unload_LLM.stderr)


"""
Argo Gateway API call for OpenAI LLMs, GPT-class

Only accessible from within Argonne National Laboratory network.
"""
def argo_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt

    if verifyprompt:
        _tok = lassi.token_analysis.token_count(system_prompt + " " + content_prompt)
        print(f"\n=====> VERIFY THE PROMPTING <===== [tokens: {_tok:,}]\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
    
    # Data to be sent as a POST in JSON format
    # Claude models do not accept top_p alongside temperature — omit it for those models.
    data = {
        "user": argo_username,
        "model": model,   # gpt35, gpt4turbo
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop": [],
        "temperature": round(temp, 1),
    }
    if not model.lower().startswith("claude"):
        data["top_p"] = 0.9

    # Convert the dict to JSON string for curl
    payload = json.dumps(data)

    response = lassi.argostream_curl.argo_fetch_with_curl(payload)

    return response


def openai_compatible_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    """
    Generic OpenAI-compatible chat-completions backend — works against OpenAI itself,
    or any self-hosted server implementing the same REST contract (vLLM, llama.cpp
    server, LM Studio, OpenRouter, etc). Connection details come from the module-level
    openai_base_url / openai_api_key globals, set by run_lassi.py from config.yaml's
    llm.openai_compatible block.
    """
    if verifyprompt:
        _tok = lassi.token_analysis.token_count(system_prompt + " " + content_prompt)
        print(f"\n=====> VERIFY THE PROMPTING <===== [tokens: {_tok:,}]\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt},
        ],
        "temperature": round(temp, 1),
    }

    url = openai_base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    response_text = resp.json()["choices"][0]["message"]["content"]

    print(response_text)

    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()

    return cleaned_response


def call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False, stream=True):
    """
    Single dispatch chokepoint for every LLM backend. llm_source is one of
    "argo", "ollama", "openai_compatible" (set via config.yaml's llm.backend field).
    """
    if llm_source == "argo":
        return argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        if stream:
            return ollama_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
        return ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "openai_compatible":
        return openai_compatible_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    else:
        raise ValueError(f"Unknown llm_source: {llm_source!r} (expected 'argo', 'ollama', or 'openai_compatible')")
