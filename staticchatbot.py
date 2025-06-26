#!/usr/bin/env python3
"""
Interactive Poem Generator with Gemini API
"""

from google import genai

# Setup API with error handling
print("=== Poem Generator ===")
print("Get your API key from: https://aistudio.google.com/app/apikey")
api_key = input("Enter your Gemini API key: ").strip()

if not api_key:
    print("Error: API key cannot be empty!")
    exit()

try:
    client = genai.Client(api_key=api_key)
    # Test the API key
    test_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello"
    )
    print("✓ API key is valid!")
except Exception as e:
    print(f"Error: Invalid API key or connection issue - {e}")
    exit()

def generate_poem(inspiration):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Write a poem inspired by: {inspiration}"
        )
        return response.text
    except Exception as e:
        print(f"Error generating poem: {e}")
        return None

def transform_poem(poem, choice):
    transformations = {
        "1": f"Shorten this poem:\n{poem}",
        "2": f"Expand this poem with more detail:\n{poem}",
        "3": f"Rewrite this poem in a different style:\n{poem}",
        "4": f"Change the tone of this poem:\n{poem}"
    }
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=transformations[choice]
        )
        return response.text
    except Exception as e:
        print(f"Error transforming poem: {e}")
        return None

# Part 1: Generate poem
inspiration = input("\nEnter a word or sentence for inspiration: ")
print("Generating poem...")

poem = generate_poem(inspiration)
if not poem:
    print("Failed to generate poem. Exiting.")
    exit()

print("\nYour poem:")
print("-" * 40)
print(poem)
print("-" * 40)

# Part 2: Transform poem
while True:
    print("\nTransformation options:")
    print("1. Shorten poem")
    print("2. Expand poem") 
    print("3. Different style")
    print("4. Change tone")
    print("5. Exit")
    
    choice = input("Choose (1-5): ")
    
    if choice == "5":
        print("Goodbye!")
        break
    elif choice in ["1", "2", "3", "4"]:
        print("Transforming...")
        new_poem = transform_poem(poem, choice)
        if new_poem:
            poem = new_poem
            print("\nTransformed poem:")
            print("-" * 40)
            print(poem)
            print("-" * 40)
    else:
        print("Invalid choice! Please enter 1-5.")