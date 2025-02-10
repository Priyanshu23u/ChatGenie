# ChatGenie

ChatGenie is an AI-powered chatbot application developed using Python. It leverages machine learning models to understand user inputs and generate appropriate responses, providing an interactive conversational experience.

## Features

- **Interactive Conversations**: Engage in real-time dialogues with the chatbot.
- **Machine Learning Integration**: Utilizes trained models for natural language understanding and response generation.
- **Extensible Design**: Easily adaptable for various conversational applications.

## Installation

### Clone the Repository:

```bash
git clone https://github.com/Priyanshu23u/ChatGenie.git
cd ChatGenie
```

### Set Up a Virtual Environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Application:

```bash
python app.py
```

Open your preferred interface (e.g., command line, web interface) to start chatting with the bot.

## Model Training

If you wish to train the chatbot model with new data:

### Prepare Training Data:

- Ensure your training data is in the correct format (e.g., JSON, CSV) and contains relevant conversational pairs.

### Train the Model:

```bash
python main.py
```

This script will process the training data and save the trained model as `chatbot_model.h5`.

## File Structure

```
ChatGenie/
│── app.py               # Main application script
│── main.py              # Script for training the chatbot model
│── chatbot_model.h5     # Pre-trained chatbot model
│── classes.pkl          # Serialized classes
│── words.pkl            # Serialized words
│── chat.json            # Training data
│── requirements.txt     # List of dependencies
```

## Dependencies

Ensure you have the following Python packages installed:

- **TensorFlow**
- **NumPy**
- **NLTK**
- **Flask**

These can be installed using the `requirements.txt` file provided.

