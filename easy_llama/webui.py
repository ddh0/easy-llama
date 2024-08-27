# webui.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

import html
import easy_llama as ez
from easy_llama.utils import assert_type
from flask import Flask, render_template, request, Response


class WebUI:

    def __init__(self, thread: ez.thread.Thread):
        assert_type(thread, ez.Thread, 'thread', 'Server')
        self.thread = thread
        self.app = Flask(__name__, static_folder='./templates', static_url_path='')
        

    def start(self, host: str, port: int):
        assert_type(host, str, 'host', 'Server.start()')
        assert_type(port, int, 'port', 'Server.start()')

        @self.app.route('/')
        def home():
            return render_template('index.html')

        @self.app.route('/submit', methods=['POST'])
        def submit():
            prompt = request.form['prompt']
            escaped_prompt = html.escape(prompt)

            def generate():
                self.thread.add_message('user', prompt)
                print(f'{ez.utils.USER_STYLE}{escaped_prompt}{ez.utils.RESET_ALL}')
                token_generator = self.thread.model.stream(
                    self.thread.inference_str_from_messages(),
                    stops=self.thread.format['stops'],
                    sampler=self.thread.sampler
                )
                response = ''
                for token in token_generator:
                    tok_text = token['choices'][0]['text']
                    response += tok_text
                    print(f'{ez.utils.BOT_STYLE}{tok_text}{ez.utils.RESET_ALL}', end='', flush=True)
                    yield tok_text
                print()
                self.thread.add_message('bot', response)

            return Response(generate(), mimetype='text/plain')
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            return '', 200
        
        self.thread.model.load()
        self.thread.warmup()

        self.app.run(host=host, port=port)
