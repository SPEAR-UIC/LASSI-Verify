import aiohttp
import asyncio
import json

async def fetch_stream():
    url = "https://localhost:5000/argoapi/api/v1/resource/streamchat/"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/plain",
        "Host": "apps-dev.inside.anl.gov"  # Updated Host for your API
    }
    
    data = {
        "user": "mdearingtest",
        "model": "gpt4o",
        "system": "You are a large language model with the name Argo.",
        "prompt": ["Tell me something interesting about quantum mechanics."],
        "stop": [],
        "temperature": 0.2,
        "top_p": 0.9
    }
    
    response = ""
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers, ssl=False) as response:
            print(f"Status: {response.status}")
            print("Streaming Response: ")

            async for chunk in response.content.iter_any():
                response += chunk.decode(errors="replace")
                print(response, end="", flush=True)
    
    return response
