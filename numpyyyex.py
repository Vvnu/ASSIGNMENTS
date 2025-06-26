import numpy as np
from textblob import TextBlob

# sum1 = 0
# array = np.array([1,2,4])
# for i in array:
#     sum1 = sum1+i
# print(sum1)
# print(array.dtype)
# print(type(array))
# array1=np.array((1,2,3))
# print(array1.sum())
# print(array1+1)
# print(array1+array)
# print(np.zeros((3,5)).dtype)
# print(np.ones((3,5)).dtype)

# print(np.eye(3 ))
# print(np.arange(1, 10, 2))
# print(np.linspace(1, 20, 5))

# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr.shape)
# print(arr[0:2, 0:2])  # Slicing the first two rows and columns



text = input("Enter a sentence: ")
blob = TextBlob(text)
sentiment = blob.sentiment

print("Sentiment Analysis:")
print("Polarity:", sentiment.polarity)
print("Subjectivity:", sentiment.subjectivity)





