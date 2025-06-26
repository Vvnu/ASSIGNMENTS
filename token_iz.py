import nltk
# Download required tokenizer model
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "Token devide string inot list of the variuos string operations"
tokens_Date = word_tokenize(text)

print(tokens_Date)
