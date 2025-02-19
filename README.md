# AllenBot - AI Chatbot for Allenhouse Group of Institutions

## Overview
AllenBot is an AI-powered chatbot designed to provide automated responses and information about Allenhouse Group of Institutions. It utilizes Natural Language Processing (NLP) techniques to understand user queries and generate relevant responses. The chatbot is built using Flask for the backend and TensorFlow for the deep learning model.

## Features
- Real-time chatbot interaction
- Pre-trained deep learning model for intent classification
- Audio response generation using gTTS (Google Text-to-Speech)
- CORS-enabled API for frontend integration
- JSON-based dataset for predefined responses
- Flask-based API with endpoints for chatbot interaction
- Hosted on a web server with support for static file handling

## Project Structure
```
├── app.py              # Flask application
├── model.py            # Model training script
├── career.json         # JSON file containing chatbot intents
├── Allenbot_model.h5   # Trained chatbot model
├── Allenbot_tokenizer.pkl  # Tokenizer for text processing
├── Allenbot_label_encoder.pkl  # Label encoder for intent classification
├── static/             # Folder for storing static files (audio responses)
```

## Technologies Used
- **Flask**: Web framework for API development
- **TensorFlow/Keras**: Deep learning framework for chatbot model
- **NLTK**: Natural Language Processing for text preprocessing
- **gTTS**: Google Text-to-Speech for generating audio responses
- **NumPy**: Data handling and processing
- **Pickle**: Model serialization
- **JSON**: Storing chatbot intents and responses

## Installation & Setup
### Prerequisites
Ensure you have Python 3.7+ installed along with the following dependencies:
```
pip install flask tensorflow numpy nltk gtts flask-cors pickle5
```

### Running the Chatbot
1. Clone the repository and navigate to the project folder.
2. Ensure the required dependencies are installed.
3. Start the Flask application:
```
python app.py
```
4. The application will run on `http://0.0.0.0:8001` by default.

## API Endpoints
### Home Route
- **Endpoint:** `/`
- **Method:** `GET`
- **Response:** Serves `index.html`

### Chatbot Prediction
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Request Body:**
  ```json
  {"message": "Hello"}
  ```
- **Response:**
  ```json
  {"response": "Hi! How can I help you?", "audio_url": "/static/response_xyz.mp3"}
  ```

## Model Training
To train or retrain the chatbot model, run:
```
python model.py
```
This will generate the `Allenbot_model.h5`, `Allenbot_tokenizer.pkl`, and `Allenbot_label_encoder.pkl` files.

## Contributions
Feel free to contribute by improving the dataset, enhancing the model, or optimizing the API.

## License
This project is licensed under the MIT License.

