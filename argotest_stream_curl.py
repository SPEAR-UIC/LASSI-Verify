import subprocess

def fetch_with_curl():
    url = "https://localhost:5000/argoapi/api/v1/resource/streamchat/"
    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Accept-Encoding: identity",
        "-H", "Cache-Control: no-cache",
        "-H", "Host: apps-dev.inside.anl.gov"
    ]
    data = '{"user": "mdearing", "model": "claudesonnet4", "system": "You are a large language model with the name Argo.", "prompt": ["Tell me something interesting about quantum mechanics."], "stop": [], "temperature": 0.2, "top_p": 0.9}'
    # data = '{"user": "mdearing", "model": "gpto3mini", "messages": [{"role": "system", "content": "You are a large language model with the name Argo"}, {"role": "user", "content": "What is your name?"}], "stop": [], "temperature": 0.2, "top_p": 0.9}'

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

# Run the function and store the output
# output = fetch_with_curl()

# print("\nFinal Output Captured:")
# print(output)  # Full response captured
