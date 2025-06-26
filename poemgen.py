from google import genai
client = genai.Client(api_key="AIzaSyDCfDH2DIqhxb200xKnQI7_NzLzhmEFZpM")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    content = "Write a short poem on the basis of the word entered by user"
)

print(response.text)

