class Handler():
    def get(self):
        return {
            'resultStatus': 'SUCCESS',
            'message': "Hello Api Handler"
        }


    def post(self,data):
        message = "Your Message Requested: {}"
        final_ret = {"status": "Success", "message": message}

        return final_ret