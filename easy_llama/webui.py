# webui.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

import os
import sys
import json

import easy_llama as ez
from easy_llama.utils import assert_type

from flask import Flask, render_template, request, Response


GREEN = ez.utils.USER_STYLE
BLUE = ez.utils.BOT_STYLE
YELLOW = ez.utils.SPECIAL_STYLE
RED = ez.utils.ERROR_STYLE
RESET = ez.utils.RESET_ALL

WARNING = \
f"""{RED}
###############################################################################
{RESET}
                          please keep in mind

            the easy-llama WebUI is not meant for production use

                   it is only intended for personal use

         the connection between client and server is not encrypted

                 and all messages are sent in cleartext

     if you expose the WebUI to the internet, you do so at your own risk

                          you have been warned
{RED}
###############################################################################
{RESET}"""


class WebUI:

    def __init__(self, thread: ez.thread.Thread):
        assert_type(thread, ez.Thread, 'thread', 'WebUI')
        self.thread = thread
        self._cancel_flag = False
        _assets_folder = os.path.join(os.path.dirname(__file__), 'assets')
        self.app = Flask(
            __name__,
            static_folder=_assets_folder,
            template_folder=_assets_folder,
            static_url_path=''
        )
    

    def _log(self, text: str) -> None:
        print(f'easy_llama: WebUI {self.thread.uuid}: {text}', file=sys.stderr, flush=True)
    

    def _get_context_string(self) -> str:
        thread_len_tokens = self.thread.len_messages()
        max_ctx_len = self.thread.model.context_length
        return f"{thread_len_tokens} / {max_ctx_len} tokens used"
    

    def start(self, host: str, port: int = 8080):
        """
        - host `str`:
            The local IP address from which to host the WebUI. For example,
            `'127.0.0.0.1'` or `'10.0.0.140'`
        - port `int`:
            The port on which to host the WebUI. Defaults to `8080`
        """
        assert_type(host, str, 'host', 'WebUI.start')
        assert_type(port, int, 'port', 'WebUI.start')
        print(WARNING, file=sys.stderr, flush=True)
        self._log(f"starting server at {host}:{port}")

        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/cancel', methods=['POST'])
        def cancel():
            print('', file=sys.stderr)
            self._log('hit cancel endpoint')
            self._cancel_flag = True
            return '', 200

        @self.app.route('/submit', methods=['POST'])
        def submit():
            self._log('hit submit endpoint')
            if self._cancel_flag:
                self._log('refuse to continue submission because cancel flag is set')
                return '', 418
            prompt = request.form['prompt']
            if prompt in ['', None]:
                self._log('do not submit empty prompt')
                return '', 418

            def generate():
                self.thread.add_message('user', prompt)
                print(f'{GREEN}{prompt}{RESET}', file=sys.stderr)
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
                        print(f'{BLUE}{tok_text}{RESET}', end='', flush=True, file=sys.stderr)
                        yield tok_text
                    else:
                        print('', file=sys.stderr)
                        self._log('cancel generation. teapot')
                        self._cancel_flag = False # reset flag
                        return '', 418 # I'm a teapot
                print('', file=sys.stderr)
                self.thread.add_message('bot', response)

            if prompt not in ['', None]:
                return Response(generate(), mimetype='text/plain')
            return '', 200
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            self._log(f"thread with UUID '{self.thread.uuid}' was reset")
            return '', 200
        
        @self.app.route('/get_context_string', methods=['GET'])
        def get_context_string():
            return json.dumps({'text': self._get_context_string()}), 200, {'ContentType': 'application/json'}
        
        @self.app.route('/remove', methods=['POST'])
        def remove():
            if len(self.thread.messages) >= 1:
                self.thread.messages.pop(-1)
                self._log('removed last message')
                return '', 200
            else:
                self._log('no previous message to remove. teapot')
                return '', 418 # I'm a teapot
        
        @self.app.route('/trigger', methods=['POST'])
        def trigger():
            self._log('hit trigger endpoint')
            if self._cancel_flag:
                self._log('refuse to trigger because cancel flag is set')
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
                        self._log('cancel generation. teapot')
                        self._cancel_flag = False # reset flag
                        return '', 418 # I'm a teapot
                print('', file=sys.stderr)
                self.thread.add_message('bot', response)

            return Response(generate(), mimetype='text/plain')
        
        self._log('loading model')
        self.thread.model.load()
        self._log('warming up thread')
        self.thread.warmup()
        
        try:
            self._log('now running Flask')
            self.app.run(
                host=host,
                port=port
            )
        except Exception as exc:
            self._log(f'{RED}exception in Flask, unloading model now{RESET}')
            self.thread.model.unload()
            raise exc
        
        print('', file=sys.stderr)
        self._log('Flask server stopped')
        self._log('----- final thread stats -----')
        self.thread.print_stats()
        self._log('------------------------------')
        self.thread.model.unload()
