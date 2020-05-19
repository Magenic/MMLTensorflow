import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, jsonify, request

MODEL_PATH='covid_19.tflite'

app = Flask(__name__)

@app.route("/covid19", methods=["POST"])
def calculateDeaths():
    print(request)
    cases = request.json['cases']

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array([[(cases /100000)**3]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    data = {'cases' : cases, 'deaths': output_data[0][0]}
    
    return jsonify(str(data))

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=8080, debug=False)