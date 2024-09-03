# webui.py
# https://github.com/ddh0/easy-llama/

"""The easy-llama WebUI"""

import os
import sys
import json
import base64

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

        the easy-llama WebUI is not guaranteed to be secure whatsoever

             it is not intended to be exposed to the internet

     if you expose the WebUI to the internet, you do so at your own risk

                          you have been warned

{RED}
###############################################################################
{RESET}"""

USING_SELF_SIGNED_SSL_CERT_WARNING = \
f"{YELLOW}you have set `ssl=True` which tells the WebUI to generate and " + \
f"use a self-signed SSL certificate. this enables secure communication " + \
f"between client and server, but your browser will probably show you a " + \
f"scary warning about a self-signed certificate. you may safely ignore " + \
f"this warning and proceed to the WebUI.{RESET}"


def _newline() -> None:
    print('', end='\n', file=sys.stderr, flush=True)


def encode(text):
    data = (text).encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def decode(base64_str):
    return base64.b64decode(base64_str).decode('utf-8')


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
        # variables used for logging to console
        self._log_host = None
        self._log_port = None
    

    def _log(self, text: str) -> None:
        if any(i is None for i in [self._log_host, self._log_port]):
            ez.utils.print_verbose(text)
        else:
            print(
                f'easy_llama: WebUI @ '
                f'{YELLOW}{self._log_host}{RESET}:'
                f'{YELLOW}{self._log_port}{RESET}: {text}',
                file=sys.stderr,
                flush=True
            )
        

    def _get_context_string(self) -> str:
        thread_len_tokens = self.thread.len_messages()
        max_ctx_len = self.thread.model.context_length
        return f"{thread_len_tokens} / {max_ctx_len} tokens used"
    

    def start(self, host: str, port: int = 8080, ssl: bool = False):
        """
        - host `str`:
            The local IP address from which to host the WebUI. For example,
            `'127.0.0.0.1'` or `'10.0.0.140'`
        - port `int`:
            The port on which to host the WebUI. Defaults to `8080`
        - ssl `bool`:
            Whether to generate and use a self-signed SSL certificate to
            encrypt traffic between client and server
        """
        print(WARNING, file=sys.stderr, flush=True)
        assert_type(host, str, 'host', 'WebUI.start')
        assert_type(port, int, 'port', 'WebUI.start')
        self._log_host = None
        self._log_port = None
        self._log(f"starting WebUI instance:")
        self._log(f"   thread.uuid           == {self.thread.uuid}")
        self._log(f"   host                  == {host}")
        self._log(f"   port                  == {port}")
        self._log_host = host
        self._log_port = port

        @self.app.route('/')
        def home():
            return render_template('index.html')
        
        @self.app.route('/cancel', methods=['POST'])
        def cancel():
            _newline()
            self._log('hit cancel endpoint - flag is set')
            self._cancel_flag = True
            return '', 200

        @self.app.route('/submit', methods=['POST'])
        def submit():
            self._cancel_flag = False
            self._log('hit submit endpoint')
            prompt = decode(request.data)
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
                        yield encode(tok_text)
                    else:
                        print(file=sys.stderr)
                        self._log('cancel generation from /submit. teapot')
                        self._cancel_flag = False # reset flag
                        return '', 418 # I'm a teapot
                _newline()
                self.thread.add_message('bot', response)

            if prompt not in ['', None]:
                return Response(generate(), mimetype='text/plain')
            return '', 200
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            self._log(f"thread was reset")
            return '', 200
        
        @self.app.route('/get_context_string', methods=['GET'])
        def get_context_string():
            return json.dumps(
                {
                    'text': encode(self._get_context_string())
                }
            ), 200, {'ContentType': 'application/json'}
        
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
            self._cancel_flag = False
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
                        yield encode(tok_text)
                    else:
                        print()
                        self._log('cancel generation from /trigger. teapot')
                        self._cancel_flag = False # reset flag
                        return '', 418 # I'm a teapot
                print('', file=sys.stderr)
                self.thread.add_message('bot', response)

            return Response(generate(), mimetype='text/plain')
        
        if not self.thread.model.is_loaded():
            self._log('loading model')
            self.thread.model.load()
        else:
            self._log('model is already loaded')
        
        self._log('warming up thread')
        self.thread.warmup()

        if ssl:
            ez.utils.print_verbose(USING_SELF_SIGNED_SSL_CERT_WARNING)
        
        try:
            self._log('now running Flask')
            self.app.run(
                host=host,
                port=port,
                ssl_context='adhoc' if ssl else None
            )
        except Exception as exc:
            self._log(f'{RED}exception in Flask: {exc}{RESET}')
            raise exc
        
        self._log_host = None
        self._log_port = None
        _newline()
