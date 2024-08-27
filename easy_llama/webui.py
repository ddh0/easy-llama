# server.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

from easy_llama.model import Model
from easy_llama.thread import Thread
from easy_llama.utils import assert_type
from flask import Flask, render_template, request, Response


class WebUI:

    def __init__(self, model: Model, thread: Thread):
        assert_type(model, Model, 'model', 'Server')
        assert_type(thread, Thread, 'thread', 'Server')
        self.model = model
        self.thread = thread
        self.app = Flask(__name__)
        

    def start(self, host: str, port: int):
        assert_type(host, str, 'host', 'Server.start()')
        assert_type(port, int, 'port', 'Server.start()')

        @self.app.route('/')
        def home():
            return render_template('index.html')

        @self.app.route('/submit', methods=['POST'])
        def submit():
            prompt = request.form['prompt']

            def generate():
                self.thread.add_message('user', prompt)
                token_generator = self.thread.model.stream(
                    self.thread.inference_str_from_messages(),
                    stops=self.thread.format['stops'],
                    sampler=self.thread.sampler
                )
                response = ''
                for token in token_generator:
                    response += token['choices'][0]['text']
                    yield token['choices'][0]['text'].encode('utf-8')
                self.thread.add_message('bot', response)

            return Response(generate(), mimetype='application/octet-stream')
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            return '', 204  # Return a 204 No Content status
        
        self.model.load()
        self.thread.warmup()

        self.app.run(host=host, port=port)
