import sys
from flask import Flask,request,jsonify, Response
from flask_cors import CORS, cross_origin
from web_interaction_sqlova import WebInteractiveParser

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = 'Content-Type'

interactive_obj = WebInteractiveParser()

@app.route("/")
def helloWorld():
    return "Hello, cross-origin-world!"

@app.route('/init',methods=['GET'])
@cross_origin()
def init():
    msg = interactive_obj.setup()
    print("inside init, msg={}".format(msg))
    return jsonify(msg)


@app.route('/proceed',methods=['GET'])
def proceed():
    user = interactive_obj.user
    input_item = interactive_obj.input_item
    g_sql = interactive_obj.g_sql
    hyp = interactive_obj.hyp
    msg = interactive_obj.agent.real_user_interactive_parsing_session(
            user, input_item, g_sql, hyp, bool_verbal=False)
    print("inside init, msg={}".format(msg))
    return jsonify(msg)

@app.route('/ask_question',methods=['GET'])
@cross_origin()
def ask_question():
    msg = interactive_obj.verified_qa()
    print("ask_question, msg={}".format(msg))
    return jsonify(msg)

@app.route('/give_options',methods=['GET'])
@cross_origin()
def ask_selection():
    msg = interactive_obj.get_selection()
    print("ask_question, msg={}".format(msg))
    return jsonify(msg)

@app.route('/response',methods=['POST'])
def start_post():
    input_json = request.get_json(force=True)
    if "init_done" in input_json:
        if input_json['init_done'] == False:
            init()
    elif "q_ans" in input_json:
        if interactive_obj.quesType == 'qa':
            if input_json['q_ans'] not in {'yes', 'no', 'exit', 'y', 'n', 'undo'}:
                ask_question()
            else:
                response = input_json['q_ans']
                if response == 'y':
                    response = 'yes'
                elif response == 'n':
                    response = 'no'
                msg = interactive_obj.use_feedback(response)
                print("options to give: {}".format(msg))
                return jsonify(msg)
        elif interactive_obj.quesType == 'use_fbk':            
            response = input_json['q_ans']
            msg = interactive_obj.user_selection(response)
            print("options to give: {}".format(msg))
            return jsonify(msg)
    return {}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091, debug=True)
