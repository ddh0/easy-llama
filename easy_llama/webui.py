# webui.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

import os
import sys
import json
import html
from cryptography.fernet import Fernet

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


def generate_key() -> bytes:
    return Fernet.generate_key()


def encrypt(key: bytes, message: str) -> bytes:
    cipher_suite = Fernet(key)
    encrypted_message = cipher_suite.encrypt(message.encode())
    return encrypted_message


def decrypt(key: bytes, encrypted_message: bytes) -> str:
    cipher_suite = Fernet(key)
    decrypted_message = cipher_suite.decrypt(encrypted_message)
    return decrypted_message.decode()


def _log(text: str) -> None:
    print(f'easy_llama: WebUI: {text}', file=sys.stderr, flush=True)


class WebUI:

    def __init__(self, thread: ez.thread.Thread):
        assert_type(thread, ez.Thread, 'thread', 'Server')
        self.thread = thread
        self._cancel_flag = False
        self._assets_folder = os.path.join(os.path.dirname(__file__), 'assets')
        self.app = Flask(
            __name__,
            static_folder=self._assets_folder,
            template_folder=self._assets_folder,
            static_url_path=''
        )
        self._session_key = generate_key()
    

    def _get_stats_string(self) -> str:
        thread_len_tokens = self.thread.len_messages()
        max_ctx_len = self.thread.model.context_length
        return f"{thread_len_tokens} / {max_ctx_len} tokens used"
    

    def start(self, host: str, port: int = 8080):
        """
        - host `str`:
            The local IP address from which to host the WebUI. For example,
            `'127.0.0.0.1'` or `'0.0.0.0'`
        - port `int`:
            The local port from which to host the WebUI. Defaults to `8080`
        """
        assert_type(host, str, 'host', 'Server.start()')
        assert_type(port, int, 'port', 'Server.start()')
        print(_WEBUI_WARNING, file=sys.stderr, flush=True)
        _log(f"starting from thread '{self.thread.uuid}' at {host}:{port}")

        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/cancel', methods=['POST'])
        def cancel():
            print()
            _log('hit cancel endpoint')
            self._cancel_flag = True
            return '', 200

        @self.app.route('/submit', methods=['POST'])
        def submit():
            _log('hit submit endpoint')
            if self._cancel_flag:
                _log('refuse to continue submission because cancel flag is set')
                return '', 418
            prompt = request.form['prompt']
            if prompt in ['', None]:
                _log('do not submit empty prompt')
                return '', 418
            escaped_prompt = html.escape(prompt)

            def generate():
                self.thread.add_message('user', prompt)
                print(f'{GREEN}{prompt}{RESET}')
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
                        print()
                        _log('cancel generation. teapot')
                        self._cancel_flag = False # reset flag
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
        
        @self.app.route('/get_stats', methods=['GET'])
        def get_placeholder_text():
            return json.dumps({'text': self._get_stats_string()}), 200, {'ContentType': 'application/json'}
        
        @self.app.route('/remove', methods=['POST'])
        def remove():
            if len(self.thread.messages) >= 1:
                self.thread.messages.pop(-1)
                _log('removed last message')
                return '', 200
            else:
                _log('no previous message to remove. teapot')
                return '', 418 # I'm a teapot
        
        @self.app.route('/trigger', methods=['POST'])
        def trigger():
            _log('hit trigger endpoint')
            if self._cancel_flag:
                _log('refuse to trigger because cancel flag is set')
                return '', 418
            def generate():
                token_generator = self.thread.model.stream(
                    self.thread.inf_str(),
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
                        print()
                        _log('cancel generation. teapot')
                        self._cancel_flag = False # reset flag
                        return '', 418 # I'm a teapot
                print()
                self.thread.add_message('bot', response)

            return Response(generate(), mimetype='text/plain')
        
        _log('loading model')
        self.thread.model.load()
        _log('warming up thread')
        self.thread.warmup()

        if host in ['0.0.0.0']:
            print(_EXPOSED_HOST_WARNING, file=sys.stderr, flush=True)
        
        try:
            _log('now running Flask')
            self.app.run(
                host=host,
                port=port
            )
        except Exception as exc:
            _log(f'{RED}exception in WebUI.app.run(), unloading model now{RESET}')
            self.thread.model.unload()
            raise exc
