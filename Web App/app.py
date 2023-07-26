import pickle
from flask import Flask, request, jsonify, render_template


# routes 
# / get - return the form template
# /predict post - return the prediction

app = Flask(__name__)

# load the model
model = pickle.load(open('model.pkl', 'rb'))

# load the vectorizer
vectorizer = pickle.load(open('feature_extraction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
     # get the data from the POST request
     data = request.get_json(force=True)
     text = [data['text']]

     # vectorize the text
     text_vectorized = vectorizer.transform(text)

     # predict the class
     prediction = model.predict(text_vectorized)

     if(prediction[0] == 1):
      response = {
          'error': False,
          'prediction': "Ham"
      }
     else:
      response = {
          'error': False,
          'prediction': "Spam"
      }
     return jsonify(response)
    except:
     response = {
          'error': True,
          'message': "An error occured"
      }
     return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
