from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

# Define custom layers
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # Implement your custom attention logic here
        return inputs

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GetItem(Layer):
    def __init__(self, index=None, **kwargs):
        super(GetItem, self).__init__(**kwargs)
        self.index = index

    def call(self, x):
        return tf.gather(x, indices=self.index, axis=1) if self.index is not None else x

    def get_config(self):
        config = super(GetItem, self).get_config()
        config.update({"index": self.index})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Flask app
app = Flask(__name__)

# Load the model with custom layers
custom_objects = {'AttentionLayer': AttentionLayer, 'GetItem': GetItem}
model = tf.keras.models.load_model('lstm_attention_model.h5', custom_objects=custom_objects)

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts([text])  # You should have a pre-trained tokenizer for your model
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    return padded

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    preprocessed_text = preprocess_text(input_text)
    
    # Predict sentiment/emotion
    prediction = model.predict(preprocessed_text)
    sentiment = np.argmax(prediction, axis=1)[0]
    
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
