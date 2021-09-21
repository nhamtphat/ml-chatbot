import reading

### START FLASK
from flask import Flask
from flask import request

app = Flask(__name__)

import reading

@app.route("/", methods = ['POST'])
def hello():
    message = request.form.get('message')
    return {'response': reading.response(message)}

if __name__ == "__main__":
    app.run(host='0.0.0.0')
