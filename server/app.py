from flask import Flask,request,jsonify
from server.services.basic import Handler
app = Flask(__name__)

handler = Handler()
@app.route('/start',methods=['POST','GET'])
def start():
    return handler.get()

@app.route('/post',methods=['POST'])
def start_post():
    input_json = request.get_json(force=True)
    dictToReturn = {'text': input_json['text']+"sdfgsdf"}
    return jsonify(dictToReturn)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091, debug=True)
