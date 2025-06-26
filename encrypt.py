user_input = input("You: ").lower()
encrypted = ""
shift = 2  # Number of bits to shift left

# Encrypting using left bit shift
for char in user_input:
    shifted = ord(char) << shift
    encrypted_char = chr(shifted % 1114111)  # Limiting within Unicode range
    encrypted += encrypted_char

print("Encrypted:", encrypted)

# Decryption using right bit shift
decrypted = ""

for char in encrypted:
    shifted = ord(char) >> shift
    decrypted_char = chr(shifted)
    decrypted += decrypted_char

print("Decrypted:", decrypted)
