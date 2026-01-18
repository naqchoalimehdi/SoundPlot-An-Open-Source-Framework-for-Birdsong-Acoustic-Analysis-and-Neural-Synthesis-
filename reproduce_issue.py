
import requests
import os

url = 'http://localhost:5000/api/analyze'
files = {'audio': open(r'data/raw/birdsong_fast.wav', 'rb')}

try:
    print("Sending request...")
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        print("Response received.")
        original_points = data.get('original_points', [])
        print(f"Number of original points: {len(original_points)}")
        if len(original_points) == 0:
            print("ISSUE REPRODUCED: original_points is empty.")
        else:
            print("Points found. Issue NOT reproduced with this file.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Exception: {e}")
finally:
    files['audio'].close()
