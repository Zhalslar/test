import requests

def test_gpt_proxy(api_url):
    # Simple proxy test
    try:
        response = requests.get(api_url)
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gpt_proxy("https://api.openai.com/v1/models")