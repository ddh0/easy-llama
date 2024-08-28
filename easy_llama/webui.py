# webui.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

import sys
import json
import html

import easy_llama as ez

from flask import Flask, render_template, request, Response

assert_type = ez.utils.assert_type

GREEN = ez.utils.USER_STYLE
BLUE = ez.utils.BOT_STYLE
YELLOW = ez.utils.SPECIAL_STYLE
RED = ez.utils.ERROR_STYLE
RESET = ez.utils.RESET_ALL

_WEBUI_WARNING = \
"""
################################################################

    the easy-llama WebUI is not meant for production use

                  it may be insecure

################################################################
"""

_EXPOSED_HOST_WARNING = \
"""
================================================================

    you are hosting from 0.0.0.0 which exposes the WebUI to    
              other devices over the network 

================================================================
"""


def _log(text: str) -> None:
    print(f'easy_llama: WebUI: {text}', file=sys.stderr, flush=True)


class WebUI:

    def __init__(self, thread: ez.thread.Thread):
        assert_type(thread, ez.Thread, 'thread', 'Server')
        self.thread = thread
        self.app = Flask(__name__, static_folder='./assets', static_url_path='')
        self._cancel_flag = False
    

    def _get_stats_string(self) -> str:
        thread_len_tokens = self.thread.len_messages()
        max_ctx_len = self.thread.model.context_length
        c = (thread_len_tokens/max_ctx_len) * 100
        ctx_used_pct = int(c) + (c > int(c)) # round up to next integer
        _fn = self.thread.model.filename
        meta_name = self.thread.model.metadata['general.name']
        model_name = _fn if meta_name in ['', None] else meta_name
        bpw = f"{self.thread.model.bpw:.2f}"
        model_display_str = f"{model_name} @ {bpw} bits per weight"
        return (
            f"{model_name}: {thread_len_tokens} / {max_ctx_len} tokens used"
        )
    

    def start(self, host: str, port: int = 8080):
        assert_type(host, str, 'host', 'Server.start()')
        assert_type(port, int, 'port', 'Server.start()')
        print(_WEBUI_WARNING, file=sys.stderr, flush=True)
        _log(f"starting from thread '{self.thread.uuid}' at {host}:{port}")

        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/cancel', methods=['POST'])
        def cancel():
            _log('hit cancel endpoint')
            self._cancel_flag = True
            return '', 200

        @self.app.route('/submit', methods=['POST'])
        def submit():
            prompt = request.form['prompt']
            escaped_prompt = html.escape(prompt)

            def generate():
                self.thread.add_message('user', prompt)
                print(f'{GREEN}{escaped_prompt}{RESET}')
                token_generator = self.thread.model.stream(
                    self.thread.inference_str_from_messages(),
                    stops=self.thread.format['stops'],
                    sampler=self.thread.sampler
                )
                response = ''
                for token in token_generator:
                    if not self._cancel_flag:
                        tok_text = token['choices'][0]['text']
                        response += tok_text
                        print(f'{BLUE}{tok_text}{RESET}', end='', flush=True)
                        yield tok_text
                    else:
                        _log('cancel generation')
                        self._cancel_flag = False # reset flag
                        print()
                        return '', 418 # I'm a teapot
                print()
                self.thread.add_message('bot', response)

            if prompt not in ['', None]:
                return Response(generate(), mimetype='text/plain')
            return '', 200
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            _log(f"thread with UUID '{self.thread.uuid}' was reset")
            return '', 200
        
        @self.app.route('/get_placeholder_text', methods=['GET'])
        def get_placeholder_text():
            return json.dumps({'placeholder_text': self._get_stats_string()}), 200, {'ContentType': 'application/json'}
        
        self.thread.model.load()
        self.thread.warmup()

        if host in ['0.0.0.0']:
            print(_EXPOSED_HOST_WARNING, file=sys.stderr, flush=True)
        
        try:
            self.app.run(host=host, port=port)
        except Exception as exc:
            _log(f'{RED}exception in WebUI.app.run(), unloading model now{RESET}')
            self.thread.model.unload()
            raise exc
