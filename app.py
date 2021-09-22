import reading

### START FLASK
from flask import Flask
from flask import request
import json

app = Flask(__name__)

@app.route("/", methods = ['POST'])
def hello():
    data = json.loads(request.data)
    message = data['message']
    return {'set_attributes': {'response': reading.response(message)}}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
