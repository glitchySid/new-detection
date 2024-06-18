from flask import Flask, request, render_template 
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__, static_url_path='/static') 

# Loading Model and Tokenizer
with open('tokenizer.json', 'r') as t:
    data = json.load(t)
    tokenizer = tokenizer_from_json(data)
model = tf.keras.models.load_model('newsdetection.keras')


@app.route('/', methods=['GET', 'POST']) 
def index(): 
    if request.method == 'POST': 
        # Retrieve the text from the textarea 
        text = request.form.get('textarea') 
        print("length",len(text))
        if len(text)==0:
            return render_template('index.html', content="Please Provide News") 
        else:
            new_sequence = tokenizer.texts_to_sequences([text])
            padded_new_sequence = pad_sequences(new_sequence, maxlen=50)
            prediction = model.predict(padded_new_sequence)
            if np.argmax(prediction)==1:
                print("News is Real")
                predicted_text = "News is Real"
            else:
                print("News is Fake")
                predicted_text = "News is Fake"
            # Print the text in terminal for verification 
            print(text) 
            return render_template('index.html', content=predicted_text) 
    return render_template('index.html')
  
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
