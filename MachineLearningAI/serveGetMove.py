from flask import Flask, render_template, request, url_for
app = Flask(__name__)
import getMachineLearningMove as ML

@app.route('/getMove', methods=['POST'])
def getMove():
    content = request.get_json(force=True)
    moveList = content['moveList']
    machineLearningMove = ML.getMove(moveList)
    return machineLearningMove

app.run(host='0.0.0.0',port=5000)
