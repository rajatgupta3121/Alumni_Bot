import nltk
import string
import json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Text preprocessing functions
def LemTokens(tokens):
    return [lemmer.lemmatize(token.lower()) for token in tokens if token not in nltk.corpus.stopwords.words('english')]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Load intents file
with open('career.json', 'r') as file:
    data = json.load(file)

patterns = []
responses = []
labels = []

# Preprocess data
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(' '.join(LemNormalize(pattern)))
        responses.append(intent['responses'])  # Store all possible responses
        labels.append(intent['tag'])

# Tokenization and padding
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Model architecture
model = Sequential()
model.add(Embedding(input_dim=2000, output_dim=100, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(GlobalMaxPooling1D())  # Using GlobalMaxPooling for better feature extraction
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(padded_sequences, np.array(encoded_labels), epochs=400, validation_split=0.2, verbose=1)


model.save('Allenbot_model.h5')
with open('Allenbot_tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
with open('Allenbot_label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Model and tokenizer saved successfully.")
