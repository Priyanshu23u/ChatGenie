import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')  # Ensure 'punkt' is installed
nltk.download('wordnet')  # Ensure 'wordnet' is installed for lemmatization
nltk.download('punkt_tab')
lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
intents = json.loads(open('chat.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenize each pattern and create a list of words, documents, and classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, then remove ignored characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # Fixed from classes to words

classes = sorted(set(classes))

# Save the word and class lists for later use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open("classes.pkl", 'wb'))

# Create training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPattern = document[0]
    wordPattern = [lemmatizer.lemmatize(word.lower()) for word in wordPattern]

    # Create a bag of words
    for word in words:
        bag.append(1) if word in wordPattern else bag.append(0)

    # Create the output row
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1  # Fixed the typo here
    training.append(bag + outputRow)

# Shuffle and convert the training data into NumPy arrays
random.shuffle(training)
training = np.array(training)

# Split the data into inputs (trainX) and outputs (trainY)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation="relu"))  # Fixed the spelling
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))  # Fixed the parentheses

# Compile the model with the SGD optimizer and categorical crossentropy loss
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print('Model saved successfully')
