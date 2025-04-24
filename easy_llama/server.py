# server.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""The easy-llama FastAPI server, including an API endpoint and a WebUI"""

# XXX: This module is WIP.

import sys
import uvicorn

import easy_llama as ez

from typing              import Optional, Union, Literal
from fastapi             import FastAPI, APIRouter
from easy_llama.utils    import assert_type
from fastapi.staticfiles import StaticFiles

class Server:
    """The easy-llama FastAPI server, providing a WebUI and an API router"""

    def __init__(
        self,
        thread: 'ez.Thread',
        host: str = "127.0.0.1",
        port: int = 8080
    ):
        assert_type(thread, getattr(ez, 'Thread'), 'thread', 'Server.__init__')
        self.thread = thread
        self.host = host
        self.port = port
        self.app = FastAPI(title=f"[easy-llama.Server @ {host}:{port}]")

        # add CORS middleware for WebUI compatibility
        self.add_cors()

        # set up API router
        self.api_router = APIRouter(prefix="/api")
        self.setup_api_endpoints()

        # mount components
        self.app.include_router(self.api_router)
        # self.app.mount(
        #     "/",
        #     StaticFiles(
        #         directory="webui",
        #         html=True
        #     ),
        #     name=f"[easy-llama.Server (WebUI) @ {host}:{port}]"
        # )
    
    def log(self, text: str, level: Literal[1,2,3,4] = 1) -> None:
        ez.utils.log(f'[easy-llama.Server @ {self.host}:{self.port}] {text}', level=level)

    def add_cors(self):
        from fastapi.middleware.cors import CORSMiddleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_api_endpoints(self):

        @self.api_router.post("/send")
        async def send(content: str) -> dict:
            """Send a message in this thread as the user and return the generated response.
            This adds your message and the bot's message to the thread."""
            self.thread.messages.append({
                'role': 'user',
                'content': content
            })
            input_tokens = self.thread.get_input_ids(role='bot')
            response_toks = self.thread.llama.generate(
                input_tokens=input_tokens,
                n_predict=-1,
                stop_tokens=self.thread._stop_tokens,
                sampler_preset=self.thread.sampler_preset
            )
            response_txt = self.thread.llama.detokenize(response_toks, special=False)
            self.thread.messages.append({
                'role': 'bot',
                'content': response_txt
            })
            return {
                'role': 'bot',
                'content': response_txt,
                'n_input_tokens': len(input_tokens),
                'n_output_tokens': len(response_toks)
            }

        @self.api_router.post("/add_message")
        async def add_message(role: str, content: str) -> None:
            """Add a message to the Thread without triggering a response"""
            self.thread.add_message(role, content)

        @self.api_router.post("/trigger")
        async def trigger() -> str:
            """Trigger a new message to be generated, even without sending a message"""
            input_ids = self.thread.get_input_ids(role='bot')
            response_toks = self.thread.llama.generate(
                input_tokens=input_ids,
                n_predict=-1,
                stop_tokens=self.thread._stop_tokens,
                sampler_preset=self.thread.sampler_preset
            )
            response_txt = self.thread.llama.detokenize(response_toks, special=False)
            self.thread.messages.append({
                'role': 'bot',
                'content': response_txt
            })
            return response_txt

        @self.api_router.get("/messages")
        async def messages() -> list[dict]:
            """Get a list of all messages in this thread"""
            return self.thread.messages

        @self.api_router.post("/sampler")
        async def sampler(
            seed:                  Optional[int]              = None,
            top_k:                 Optional[int]              = None,
            top_p:                 Optional[float]            = None,
            min_p:                 Optional[float]            = None,
            xtc_probability:       Optional[float]            = None,
            xtc_threshold:         Optional[float]            = None,
            typical_p:             Optional[float]            = None,
            temp:                  Optional[float]            = None,
            dynatemp_delta:        Optional[float]            = None,
            dynatemp_exponent:     Optional[float]            = None,
            penalty_last_n:        Optional[int]              = None,
            penalty_repeat:        Optional[float]            = None,
            penalty_freq:          Optional[float]            = None,
            penalty_present:       Optional[float]            = None,
            dry_multiplier:        Optional[float]            = None,
            dry_base:              Optional[float]            = None,
            dry_allowed_length:    Optional[int]              = None,
            dry_penalty_last_n:    Optional[int]              = None,
            mirostat:              Optional[int]              = None,
            top_n_sigma:           Optional[float]            = None,
            mirostat_tau:          Optional[float]            = None,
            mirostat_eta:          Optional[float]            = None,
            dry_sequence_breakers: Optional[list[str]]        = None,
            logit_bias:            Optional[dict[int, float]] = None
        ) -> dict[str, Union[int, float, list[str], dict[int, float]]]:
            """Control the sampler settings over the API"""
            
            current = self.thread.sampler_preset
            
            # create new sampler preset with provided values or current values
            new_preset = ez.SamplerPreset(
                seed=seed if seed is not None else current.seed,
                top_k=top_k if top_k is not None else current.top_k,
                top_p=top_p if top_p is not None else current.top_p,
                min_p=min_p if min_p is not None else current.min_p,
                xtc_probability=xtc_probability if xtc_probability is not None else (
                    current.xtc_probability
                ),
                xtc_threshold=xtc_threshold if xtc_threshold is not None else (
                    current.xtc_threshold
                ),
                typical_p=typical_p if typical_p is not None else current.typical_p,
                temp=temp if temp is not None else current.temp,
                dynatemp_delta=dynatemp_delta if dynatemp_delta is not None else (
                    current.dynatemp_delta
                ),
                dynatemp_exponent=dynatemp_exponent if dynatemp_exponent is not None else (
                    current.dynatemp_exponent
                ),
                penalty_last_n=penalty_last_n if penalty_last_n is not None else (
                    current.penalty_last_n
                ),
                penalty_repeat=penalty_repeat if penalty_repeat is not None else (
                    current.penalty_repeat
                ),
                penalty_freq=penalty_freq if penalty_freq is not None else current.penalty_freq,
                penalty_present=penalty_present if penalty_present is not None else (
                    current.penalty_present
                ),
                dry_multiplier=dry_multiplier if dry_multiplier is not None else (
                    current.dry_multiplier
                ),
                dry_base=dry_base if dry_base is not None else current.dry_base,
                dry_allowed_length=dry_allowed_length if dry_allowed_length is not None else (
                    current.dry_allowed_length
                ),
                dry_penalty_last_n=dry_penalty_last_n if dry_penalty_last_n is not None else (
                    current.dry_penalty_last_n
                ),
                mirostat=mirostat if mirostat is not None else current.mirostat,
                top_n_sigma=top_n_sigma if top_n_sigma is not None else current.top_n_sigma,
                mirostat_tau=mirostat_tau if mirostat_tau is not None else current.mirostat_tau,
                mirostat_eta=mirostat_eta if mirostat_eta is not None else current.mirostat_eta,
                dry_sequence_breakers=dry_sequence_breakers if (
                    dry_sequence_breakers is not None
                ) else current.dry_sequence_breakers,
                logit_bias=logit_bias if logit_bias is not None else current.logit_bias
            )
            
            # update the current sampler preset
            self.thread.sampler_preset = new_preset
            
            # return all current values
            return {
                'seed'                  : new_preset.seed,
                'top_k'                 : new_preset.top_k,
                'top_p'                 : new_preset.top_p,
                'min_p'                 : new_preset.min_p,
                'xtc_probability'       : new_preset.xtc_probability,
                'xtc_threshold'         : new_preset.xtc_threshold,
                'typical_p'             : new_preset.typical_p,
                'temp'                  : new_preset.temp,
                'dynatemp_delta'        : new_preset.dynatemp_delta,
                'dynatemp_exponent'     : new_preset.dynatemp_exponent,
                'penalty_last_n'        : new_preset.penalty_last_n,
                'penalty_repeat'        : new_preset.penalty_repeat,
                'penalty_freq'          : new_preset.penalty_freq,
                'penalty_present'       : new_preset.penalty_present,
                'dry_multiplier'        : new_preset.dry_multiplier,
                'dry_base'              : new_preset.dry_base,
                'dry_allowed_length'    : new_preset.dry_allowed_length,
                'dry_penalty_last_n'    : new_preset.dry_penalty_last_n,
                'mirostat'              : new_preset.mirostat,
                'top_n_sigma'           : new_preset.top_n_sigma,
                'mirostat_tau'          : new_preset.mirostat_tau,
                'mirostat_eta'          : new_preset.mirostat_eta,
                'dry_sequence_breakers' : new_preset.dry_sequence_breakers,
                'logit_bias'            : new_preset.logit_bias
            }

        @self.api_router.get("/summarize")
        async def summarize() -> str:
            """Generate and return a summary of the thread content"""
            return self.thread.summarize()

        @self.api_router.get("/info")
        async def get_info() -> dict:
            """Return some info about the llama model and the thread"""
            with ez.utils.suppress_output():
                input_ids = self.thread.get_input_ids(role=None)
            n_thread_tokens = len(input_ids)
            n_ctx = self.thread.llama.n_ctx()
            c = (n_thread_tokens/n_ctx) * 100
            ctx_used_pct = int(c) + (c > int(c)) # round up to next integer
            return {
                'model_name': self.thread.llama.name(),
                'model_n_params': self.thread.llama.n_params(),
                'model_bpw': self.thread.llama.bpw(),
                'n_ctx': n_ctx,
                'n_ctx_train': self.thread.llama.n_ctx_train(),
                'n_thread_tokens': n_thread_tokens,
                'n_thread_messages': len(self.messages),
                'ctx_used_pct': ctx_used_pct
            }

        @self.api_router.post("/cancel")
        async def cancel() -> None:
            """If the model is currently generating, cancel it. Otherwise, do nothing."""
            # TODO
            pass

        @self.api_router.post("/reset")
        async def reset() -> None:
            """Reset the thread to its default state"""
            self.thread.reset()

    def run(self):
        self.log('starting uvicorn!')
        try:
            uvicorn.run(self.app, host=self.host, port=self.port)
        except Exception as exc:
            self.log(f'exception in uvicorn: {type(exc).__name__}: {exc}', 3)
            sys.exit(1)
