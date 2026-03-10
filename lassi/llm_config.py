import ollama
from ollama import ResponseError
from cerebras.cloud.sdk import Cerebras
import json
import requests
import httpx
import aiohttp
import asyncio
import subprocess
import re
import html
import time
from IPython.display import display, HTML, clear_output
import lassi.argostream_curl

import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", InsecureRequestWarning)

"""
Local hosted LLM via OLLAMA API, GGUF quantized models

Requirement: ollama serve [from server command line]
Requirement: ollama.pull('modelname') [from Python script to pre-load selected model at the server]

ollama.list() [preview what is available to use on the server]

Example: model='codestral:latest'
"""


def ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    
    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
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

    print(response_text)
    
    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()
    
    return cleaned_response  # response_text


def cerebras_llm(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    # Slow things down so we don't hit the per minute request count limit
    time.sleep(10)
    
    # Create a HTML display object with a <pre> tag for formatted text
    html_display = display(HTML(""), display_id=True)

    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
            
    response_text = ""

    client = Cerebras(
        # This is the default and can be omitted
        api_key="csk-pt8c8yt4jc28jyvmn3fj36e4ty8tmr8xm5d6e52e4ht92698"  # os.environ.get("CEREBRAS_API_KEY")
    )

    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt}
        ],
        model=model,
        stream=True,
        max_completion_tokens=65536,
        temperature=temp,
        top_p=0.9,
        # reasoning_effort="medium"
    )

    # Process each chunk in the stream
    try:
        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content or ""
            response_text += chunk_content
    
            # Escape HTML and highlight <think> tags
            escaped_text = html.escape(response_text)
    
            # Style opening tags
            styled_text = re.sub(
                r'&lt;think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
                escaped_text
            )
            # Style closing tags
            styled_text = re.sub(
                r'&lt;/think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
                styled_text
            )
            
            # Create typewriter effect with cursor
            display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"  
            # Put everything on one line to avoid extra whitespace
            html_content = (
                f"<pre class='lassi-out' style='padding:10px;border-radius:5px;"
                f"white-space:pre-wrap;font-family:monospace;margin:0;'>{display_text}</pre>"
                f"<style>"
                f".lassi-out{{background:#f0f0f0;color:#1e1e1e;border:1px solid #ccc;}}"
                f"@media(prefers-color-scheme:dark){{.lassi-out{{background:#1e1e1e;color:#d4d4d4;border:1px solid #444;}}}}"
                f"@keyframes blink {{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}"
                f"</style>"
            )
        
        # Update the display output with styled text
        html_display.update(HTML(html_content))
        
    except ResponseError as e:
        if "unexpected EOF" in str(e):
            # Log the event, check if the output seems complete, and continue
            print("Stream ended unexpectedly but continuing with received output.")
        else:
            raise
    
    # Clear the streaming display after completion
    # clear_output(wait=True)

    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()
    
    return cleaned_response  # response_text

    
def ollama_llm(model, num_ctx, system_prompt, content_prompt, temp=0.2, verifyprompt=False):
    # Create an HTML display object with a <pre> tag for formatted text
    html_display = display(HTML(""), display_id=True)

    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
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
            response_text += chunk_content
    
            # Escape HTML and highlight <think> tags
            escaped_text = html.escape(response_text)
    
            # Style opening tags
            styled_text = re.sub(
                r'&lt;think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
                escaped_text
            )
            # Style closing tags
            styled_text = re.sub(
                r'&lt;/think&gt;',
                '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
                styled_text
            )
            
            # Create typewriter effect with cursor
            display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"  
            # Put everything on one line to avoid extra whitespace
            html_content = (
                f"<pre class='lassi-out' style='padding:10px;border-radius:5px;"
                f"white-space:pre-wrap;font-family:monospace;margin:0;'>{display_text}</pre>"
                f"<style>"
                f".lassi-out{{background:#f0f0f0;color:#1e1e1e;border:1px solid #ccc;}}"
                f"@media(prefers-color-scheme:dark){{.lassi-out{{background:#1e1e1e;color:#d4d4d4;border:1px solid #444;}}}}"
                f"@keyframes blink {{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}"
                f"</style>"
            )
        
        # Update the display output with styled text
        html_display.update(HTML(html_content))
        
    except ResponseError as e:
        if "unexpected EOF" in str(e):
            # Log the event, check if the output seems complete, and continue
            print("Stream ended unexpectedly but continuing with received output.")
        else:
            raise
    
    # Clear the streaming display after completion
    # clear_output(wait=True)

    # Final cleaning of the complete response
    # Conditional cleaning only if <think> tags exist
    if re.search(r'<think>.*?</think>', response_text, flags=re.DOTALL):
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    else:
        cleaned_response = response_text.strip()
    
    return cleaned_response  # response_text


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
async def fetch_stream(url, data, headers):
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers, ssl=False) as response:
            print(f"Status: {response.status}")
            print("Streaming Response: ")

            async for chunk in response.content.iter_any():
                text = chunk.decode(errors="replace")
                print(text, end="", flush=True)  # Print the text live
                full_response += text  # Store the full response
                # print(chunk.decode(errors="replace"), end="", flush=True)

    return full_response


def fetch_with_curl(url, data):

    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept-Encoding: identity",
        "-H", "Cache-Control: no-cache",
        "-H", "Host: apps-dev.inside.anl.gov"
    ]

    # Run curl as a subprocess, capturing output as it streams
    process = subprocess.Popen(
        ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered for real-time output
    )

    output = ""

    # Read output line by line to maintain real-time streaming
    for line in iter(process.stdout.readline, ''):
        print(line, end="", flush=True)  # Print each chunk live
        output += line  # Capture full response

    process.stdout.close()
    process.wait()  # Ensure process exits cleanly

    return output  # Store full response for later use

    
def argo_llm(model, system_prompt, content_prompt, temp=0.2, verifyprompt=False, uri="remote"):
    # Argo Gateway API endpoint to POST
    if uri == "local":
        url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"
    else:
        url = "https://localhost:5000/argoapi/api/v1/resource/streamchat/"
    # url = "https://localhost:5000/argoapi/api/v1/resource/chat/"
    # url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"
    
    
    if model == "gpto1preview":
        content_prompt = system_prompt + "\n\n" + content_prompt
    
    if verifyprompt:
        print("\n=====> VERIFY THE PROMPTING <=====\n")
        print(f"TEMPERATURE: {round(temp, 1)}")
        print(f"\nSYSTEM: {system_prompt}")
        print(f"\nPROMPT: {content_prompt}")
        print("\n=====^^^^^ END PROMPT REVIEW ^^^^^=====\n")
    else:
        print("\n=====< LLM Prompt Not Shown >=====\n")
    
    # Data to be sent as a POST in JSON format
    data = {
        "user": "mdearing",
        "model": model,   # gpt35, gpt4turbo
        "system": system_prompt,
        "prompt": [content_prompt],
        "stop": [],
        "temperature": round(temp, 1),
        "top_p": 0.9,
    }

    # Convert the dict to JSON string for curl
    payload = json.dumps(data)

    # Add a header stating that the content type is JSON
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/plain",  # Expect standard text streaming
        "Accept-Encoding": "identity",  # Prevent gzip compression
        "Cache-Control": "no-cache",
        "Host": "apps-dev.inside.anl.gov",
    }

    response = lassi.argostream_curl.argo_fetch_with_curl(payload)
    # response = requests.post(url, json=data, headers=headers, verify=False)

    # print(response.json()['response'])

    # headers = [
    #     "-H", "Content-Type: application/json",
    #     "-H", "Accept: text/plain",
    #     "-H", "Host: apps-dev.inside.anl.gov"
    # ]

    # Run curl as a subprocess, capturing output as it streams
    # process = subprocess.Popen(
    #     ["curl", "-k", "-X", "POST", url, *headers, "-d", payload],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    #     bufsize=1  # Line-buffered for real-time output
    # )

    # response_text = []

    # Read output line by line and capture everything
    # with process.stdout:
    #     for line in iter(process.stdout.readline, ''):
    #         print(line, end="", flush=True)  # Print each chunk live
    #         response_text.append(line.strip())  # Store the full response properly

    # Ensure the process is completed
    # process.wait()

    # Join the response lines properly before returning
    # final_response = "\n".join(response_text)
    
    # Run the async function
    # response_text = fetch_stream(url, data, headers)
    #response_text = fetch_with_curl(url, payload)

    # print("response_text: \n\n", response_text)

    # Send POST request
    # response = requests.post(url, data=payload, headers=headers, stream=True)
    # response_text = response.json()['response']
    # Use httpx to send the request with streaming enabled
    # with httpx.stream("POST", url, json=data, headers=headers) as response:
    #     print("Status Code: ", response.status_code)
    #     print("Headers: ", response.headers)
    #     print("Streaming Response: ")
    #     # Read chunks as they arrive
    #     for chunk_content in response.iter_bytes():  # response.iter_content(chunk_size=1):
    #         if chunk_content:
    #        = ""      response_text += chunk_content
    #             print(chunk_content.decode(errors="replace"), end="", flush=True)  # Print each chunk as it arrives

    return response
