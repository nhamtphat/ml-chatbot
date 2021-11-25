import reading

### START FLASK
from flask import Flask
from flask import request
import json

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def index():
    return "Servcer is running"

@app.route("/chat", methods = ['GET'])
def chat():
    data = request.args
    message = data['message']
    userId = data['user_id']
    return {'set_attributes': {'response': reading.response(message, userId)}}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
