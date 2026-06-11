# Import the necessary class from NLTK
from nltk.tokenize import RegexpTokenizer
message = "Amazing event at #NLPConference_20! Over 1000 attendees from 20+ countries. #Networking #Tech"
# Convert the message to lower case 
message_lower = message.lower()
# Define a tokenizer that splits the text into words
word_tokenizer = RegexpTokenizer(r'\w+')
# Tokenize the text
words = word_tokenizer.tokenize(message_lower)
print(words)