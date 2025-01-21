# webui.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""Submodule containing the easy-llama WebUI server"""

import os
import sys
import json
import base64

from .thread import Thread
from .utils import print_info, assert_type, Colors, ez_encode, ez_decode

from cryptography import x509
from cryptography.x509.oid import NameOID
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask, render_template, request, Response
from cryptography.hazmat.primitives.serialization import Encoding

UTC = timezone.utc

RESET  = Colors.RESET
GREEN  = Colors.GREEN
YELLOW = Colors.YELLOW
RED    = Colors.RED
BLUE   = Colors.BLUE


WARNING = f"""{RED}
################################################################################
{RESET}

                          please keep in mind

        the easy-llama WebUI is not guaranteed to be secure whatsoever

             it is not intended to be exposed to the internet

     if you expose the WebUI to the internet, you do so at your own risk

                          you have been warned

{RED}
################################################################################
{RESET}"""


SSL_CERT_FIRST_TIME_WARNING = \
f"{YELLOW}you have just generated a new self-signed SSL certificate and " + \
f"key. your browser will probably warn you about an untrusted " + \
f"certificate. this is expected and you may safely proceed to the WebUI. " + \
f"subsequent WebUI sessions will re-use this SSL certificate.{RESET}"


ASSETS_FOLDER = os.path.join(os.path.dirname(__file__), 'assets')

if not os.path.exists(ASSETS_FOLDER):
    raise RuntimeError(
        f'the easy-llama/assets folder was moved or deleted, so the WebUI '
        f'cannot start. consider re-installing easy-llama.'
    )


MAX_LENGTH_INPUT = 1_000_000 # one million characters


def generate_self_signed_ssl_cert() -> None:
    """
    Generate a self-signed SSL certificate and store it in the assets folder
    """

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    public_key = private_key.public_key()

    # these values are required
    name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "XY"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "DUMMY_STATE"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "DUMMY_LOCALITY"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EZLLAMA LLC"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    builder = x509.CertificateBuilder(
        subject_name=name,
        issuer_name=name,
        public_key=public_key,
        serial_number=x509.random_serial_number(),
        not_valid_before=datetime.now(tz=UTC),
        not_valid_after=datetime.now(tz=UTC) + timedelta(days=36500),
    )

    builder = builder.add_extension(
        x509.SubjectAlternativeName([x509.DNSName("localhost")]),
        critical=False,
    )

    # sign the certificate with the private key
    certificate = builder.sign(
        private_key=private_key,
        algorithm=hashes.SHA256(),
    )

    # save key
    with open(f"{ASSETS_FOLDER}/key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    # save certificate
    with open(f"{ASSETS_FOLDER}/cert.pem", "wb") as f:
        f.write(certificate.public_bytes(Encoding.PEM))

def check_for_ssl_cert() -> bool:
    """
    Return `True` if a local SSL certificate already exists

    If not, remove any orphaned certificate files and return `False`
    """
    fns = [
        f'{ASSETS_FOLDER}/cert.pem',
        f'{ASSETS_FOLDER}/key.pem'
    ]
    if all(os.path.exists(fn) for fn in fns):
        return True
    else:
        for fn in fns:
            if os.path.exists(fn):
                os.remove(fn)
        return False

def newline() -> None:
    """Print a newline to stderr"""
    print('', end='\n', file=sys.stderr, flush=True)

def encode_transfer(text: str) -> str:
    """Encode the outgoing string into a Base64 string"""
    encoded = base64.b64encode(text.encode('utf-8', errors='strict')).decode('utf-8')
    print(f"sending encoded: {encoded!r}")
    return encoded

def decode_transfer(text: str) -> str:
    """Decode the incoming Base64 string into a regular string"""
    decoded = base64.b64decode(text).decode('utf-8', errors='strict')
    print(f"recieved decoded: {decoded!r}")
    return decoded

def assert_max_length(text: str) -> None:
    """Fail if the given text exceeds the allowed input length"""
    if len(text) > MAX_LENGTH_INPUT:
        raise AssertionError(
            f'length of input exceeds maximum allowed length of '
            f'{MAX_LENGTH_INPUT:,} characters'
        )


class WebUI:
    """
    The easy-llama WebUI server
    """

    def __init__(self, thread: Thread):
        """
        Create an instance of the WebUI server based on an existing Thread
        """
        assert_type(thread, Thread, 'thread', 'WebUI')
        self.thread = thread
        self._cancel_flag = False

        self.app = Flask(
            __name__,
            static_folder=ASSETS_FOLDER,
            template_folder=ASSETS_FOLDER,
            static_url_path=''
        )

        # variables used for logging to console
        self._log_host = None
        self._log_port = None
    

    def log(self, text: str) -> None:
        if any(i is None for i in [self._log_host, self._log_port]):
            print_info(text)
        else:
            print(
                f'easy_llama: WebUI @ '
                f'{YELLOW}{self._log_host}{RESET}:'
                f'{YELLOW}{self._log_port}{RESET}: {text}',
                file=sys.stderr,
                flush=True
            )
        

    def _get_context_string(self) -> str:
        thread_len_tokens = len(self.thread.get_input_ids(role=None))
        max_ctx_len = self.thread.llama._n_ctx

        return f"{thread_len_tokens} / {max_ctx_len} tokens used"
    

    def start(self, host: str, port: int = 8080, ssl: bool = False):
        """
        Start the WebUI on the specified address and port

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

        self.log(f"starting WebUI instance:")
        self.log(f"   host        == {host}")
        self.log(f"   port        == {port}")
        self.log(f"   ssl (HTTPS) == {ssl}")

        if ssl:

            if check_for_ssl_cert():
                self.log('re-using previously generated SSL certificate')

            else:
                self.log('generating self-signed SSL certifcate')
                generate_self_signed_ssl_cert()
                print(SSL_CERT_FIRST_TIME_WARNING, file=sys.stderr, flush=True)


        @self.app.route('/')
        def home():
            return render_template('index.html')
        

        @self.app.route('/convo', methods=['GET'])
        def convo():

            msgs_dict = dict()

            i = 0
            for msg in self.thread.messages:
                msgs_dict[i] = { encode_transfer(msg['role']) : encode_transfer(msg['content']) }
                i += 1
            
            json_convo = json.dumps(msgs_dict)
            return json_convo, 200, { 'ContentType' : 'application/json' }
        
        
        @self.app.route('/cancel', methods=['POST'])
        def cancel():
            newline()
            self.log('hit cancel endpoint - flag is set')
            self._cancel_flag = True
            return '', 200


        @self.app.route('/submit', methods=['POST'])
        def submit():
            self.log('hit submit endpoint')
            prompt = decode_transfer(request.data)

            if prompt in ['', None]:
                self.log('do not submit empty prompt')
                return '', 418

            def generate():
                self.thread.messages.append({
                    'role': 'user',
                    'content': prompt
                })
                print(f'{GREEN}{prompt}{RESET}', file=sys.stderr)
                input_ids = self.thread.get_input_ids()
                token_generator = self.thread.llama.stream(
                    input_tokens=input_ids,
                    n_predict=-1,
                    stop_tokens=self.thread.llama.eog_tokens,
                    sampler_preset=self.thread.sampler_preset
                )

                response_toks = []
                detok_bytes_buffer = b''
                response_txt = ''

                for token in token_generator:
                    if self._cancel_flag:
                        print(file=sys.stderr)
                        self.log('cancel generation from /submit. teapot')
                        self._cancel_flag = False
                        return '', 418  # I'm a teapot

                    response_toks.append(token)
                    detok_bytes_buffer += self.thread.llama.token_to_piece(token, special=False)
                    try:
                        detok_txt = detok_bytes_buffer.decode('utf-8', errors='strict')
                    except UnicodeDecodeError:
                        pass  # try again on next token
                    else:
                        detok_bytes_buffer = b''
                        response_txt += detok_txt
                        print(
                            f'{BLUE}{detok_txt}{RESET}',
                            end='',
                            flush=True,
                            file=sys.stderr
                        )
                        yield encode_transfer(detok_txt) + '\n'  # delimiter between tokens

                # print any leftover bytes (though ideally there should be none)
                if detok_bytes_buffer != b'':
                    leftover_txt = ez_decode(detok_bytes_buffer)
                    response_txt += leftover_txt
                    print(
                        f'{BLUE}{leftover_txt}{RESET}',
                        end='',
                        flush=True,
                        file=sys.stderr
                    )
                    yield encode_transfer(leftover_txt) + '\n'  # delimiter between tokens

                self._cancel_flag = False
                newline()
                self.thread.messages.append({
                    'role': 'bot',
                    'content': response_txt
                })

            if prompt not in ['', None]:
                return Response(generate(), mimetype='text/plain')

            return '', 200


        @self.app.route('/reset', methods=['POST'])
        def reset():
            self.thread.reset()
            self.log(f"thread was reset")
            return '', 200
        

        @self.app.route('/get_context_string', methods=['GET'])
        def get_context_string():

            json_content = json.dumps(
                { 'text' : encode_transfer(self._get_context_string()) }
            )

            return json_content, 200, {'ContentType': 'application/json'}
        

        @self.app.route('/remove', methods=['POST'])
        def remove():

            if len(self.thread.messages) >= 1:
                self.thread.messages.pop(-1)
                self.log('removed last message')
                return '', 200
            
            else:
                self.log('no previous message to remove. teapot')
                return '', 418 # I'm a teapot
        

        @self.app.route('/trigger', methods=['POST'])
        def trigger():
            self.log('hit trigger endpoint')
            prompt = decode_transfer(request.data)

            if prompt not in ['', None]:
                self.log(f'trigger with prompt: {prompt!r}')
                prompt_tokens = self.thread.llama.tokenize(
                    text_bytes=prompt,
                    add_special=False,
                    parse_special=False
                )
            else:
                self.log(f'trigger without prompt')
                prompt_tokens = []

            def generate():
                input_ids = self.thread.get_input_ids(role='bot') + prompt_tokens
                token_generator = self.thread.llama.stream(
                    input_tokens=input_ids,
                    n_predict=-1,
                    stop_tokens=self.thread.llama.eog_tokens,
                    sampler_preset=self.thread.sampler_preset
                )

                response_toks = []
                detok_bytes_buffer = b''
                response_txt = ''

                for token in token_generator:
                    if self._cancel_flag:
                        print()
                        self.log('cancel generation from /trigger. teapot')
                        self._cancel_flag = False  # reset flag
                        return '', 418  # I'm a teapot

                    response_toks.append(token)
                    detok_bytes_buffer += self.thread.llama.token_to_piece(token, special=False)

                    try:
                        detok_txt = detok_bytes_buffer.decode('utf-8', errors='strict')
                    except UnicodeDecodeError:
                        pass  # try again on next token
                    else:
                        detok_bytes_buffer = b''
                        response_txt += detok_txt
                        print(f'{BLUE}{detok_txt}{RESET}', end='', flush=True)
                        yield encode_transfer(detok_txt) + '\n'  # delimiter between tokens

                # print any leftover bytes (though ideally there should be none)
                if detok_bytes_buffer != b'':
                    leftover_txt = ez_decode(detok_bytes_buffer)
                    response_txt += leftover_txt
                    print(f'{BLUE}{leftover_txt}{RESET}', end='', flush=True)
                    yield encode_transfer(leftover_txt) + '\n'  # delimiter between tokens

                self._cancel_flag = False
                print('', file=sys.stderr)
                self.thread.messages.append({
                    'role': 'bot',
                    'content': prompt + response_txt
                })

            return Response(generate(), mimetype='text/plain')
        

        @self.app.route('/summarize', methods=['GET'])
        def summarize():
            summary = self.thread.summarize()
            response = encode_transfer(summary)
            self.log(f"generated summary: {BLUE}{summary!r}{RESET}")
            return response, 200, {'ContentType': 'text/plain'}
        

        @self.app.route('/settings', methods=['GET'])
        def settings():
            # TODO: update this to work with SamplerParams
            raise NotImplementedError
            return render_template('settings.html')


        @self.app.route('/update_sampler', methods=['POST'])
        def update_sampler():
            # TODO: update this to work with SamplerParams
            raise NotImplementedError
            data = request.get_json()
            self.thread.sampler.max_len_tokens = data.get('max_len_tokens', self.thread.sampler.max_len_tokens)
            self.thread.sampler.top_k = data.get('top_k', self.thread.sampler.top_k)
            self.thread.sampler.top_p = data.get('top_p', self.thread.sampler.top_p)
            self.thread.sampler.min_p = data.get('min_p', self.thread.sampler.min_p)
            self.thread.sampler.temp = data.get('temp', self.thread.sampler.temp)
            self.thread.sampler.frequency_penalty = data.get('frequency_penalty', self.thread.sampler.frequency_penalty)
            self.thread.sampler.presence_penalty = data.get('presence_penalty', self.thread.sampler.presence_penalty)
            self.thread.sampler.repeat_penalty = data.get('repeat_penalty', self.thread.sampler.repeat_penalty)
            return '', 200


        @self.app.route('/get_sampler', methods=['GET'])
        def get_sampler():
            # TODO: update this to work with SamplerParams
            raise NotImplementedError
            sampler_settings = {
                'max_len_tokens': self.thread.sampler.max_len_tokens,
                'top_k': self.thread.sampler.top_k,
                'top_p': self.thread.sampler.top_p,
                'min_p': self.thread.sampler.min_p,
                'temp': self.thread.sampler.temp,
                'frequency_penalty': self.thread.sampler.frequency_penalty,
                'presence_penalty': self.thread.sampler.presence_penalty,
                'repeat_penalty': self.thread.sampler.repeat_penalty
            }
            return json.dumps(sampler_settings), 200, {'ContentType': 'application/json'}
        
        self.log('warming up thread')
        self.thread.warmup()

        self.log('now running Flask')

        # these variables are used when printing logs
        self._log_host = host
        self._log_port = port
        
        try:
            self.app.run(
                host=host,
                port=port,
                ssl_context=(
                    f'{ASSETS_FOLDER}/cert.pem',
                    f'{ASSETS_FOLDER}/key.pem'
                ) if ssl else None
            )

        except Exception as exc:
            newline()
            self.log(f'{RED}exception in Flask: {exc}{RESET}')
            raise exc
        
        else:
            newline()
            self._log_host = None
            self._log_port = None
        
        self.log('Flask stopped')
