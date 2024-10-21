from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import json
from flask_cors import CORS
import os
from gtts import gTTS
import nltk
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences


app = Flask(__name__, static_folder='static')

model = tf.keras.models.load_model('Allenbot_model.h5')
with open('Allenbot_tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('Allenbot_label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


with open('career.json', 'r') as file:
    data = json.load(file)

max_length = 100  

def LemTokens(tokens):
    return [nltk.stem.WordNetLemmatizer().lemmatize(token.lower()) for token in tokens if token not in nltk.corpus.stopwords.words('english')]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided.'}), 400

        normalized_message = ' '.join(LemNormalize(user_message))
        sequence = tokenizer.texts_to_sequences([normalized_message])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')  

        prediction = model.predict(padded_sequence)
        intent_label = np.argmax(prediction)
        intent = label_encoder.inverse_transform([intent_label])[0]

        response = "Content not available. Could you please ask something else?"
        for intent_obj in data['intents']:
            if intent_obj['tag'] == intent:
                response = np.random.choice(intent_obj['responses'])
                break

        audio_file = os.path.join('static', 'response.mp3')
        tts = gTTS(response)
        tts.save(audio_file)

        return jsonify({'response': response, 'audio_url': f'http://localhost:8001/static/response.mp3'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host="0.0.0.0", port=8001, debug=True)
