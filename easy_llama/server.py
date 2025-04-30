# server.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""The easy-llama FastAPI server, including an API endpoint and a WebUI"""

import os
import uvicorn

import easy_llama as ez

from typing                   import Optional, Union, Literal
from fastapi                  import FastAPI, APIRouter, Body
from fastapi.middleware.cors  import CORSMiddleware
from easy_llama.utils         import assert_type
from fastapi.staticfiles      import StaticFiles
from pydantic                 import BaseModel

WEBUI_DIRECTORY = os.path.join(os.path.dirname(__file__), 'webui')

#
# Pydantic Models for FastAPI
#

class StatusResponseModel(BaseModel):
    success: bool

class MessageModel(BaseModel):
    role: str
    content: str

class MessageResponseModel(BaseModel):
    role: str
    content: str
    n_input_tokens: int
    n_output_tokens: int

class SetSysPromptRequestModel(BaseModel):
    content: str

class SummaryResponseModel(BaseModel):
    summary: str

class InfoResponseModel(BaseModel):
    model_name: str
    model_n_params: int
    model_bpw: float
    n_tokens: int
    n_ctx: int
    n_ctx_train: int
    n_thread_tokens: int
    n_thread_messages: int

class SamplerSettingsModel(BaseModel):
    seed:                  Optional[int]
    top_k:                 Optional[int]
    top_p:                 Optional[float]
    min_p:                 Optional[float]
    xtc_probability:       Optional[float]
    xtc_threshold:         Optional[float]
    typical_p:             Optional[float]
    temp:                  Optional[float]
    dynatemp_delta:        Optional[float]
    dynatemp_exponent:     Optional[float]
    penalty_last_n:        Optional[int]
    penalty_repeat:        Optional[float]
    penalty_freq:          Optional[float]
    penalty_present:       Optional[float]
    dry_multiplier:        Optional[float]
    dry_base:              Optional[float]
    dry_allowed_length:    Optional[int]
    dry_penalty_last_n:    Optional[int]
    mirostat:              Optional[int]
    top_n_sigma:           Optional[float]
    mirostat_tau:          Optional[float]
    mirostat_eta:          Optional[float]
    dry_sequence_breakers: Optional[list[str]]
    logit_bias:            Optional[dict[int, float]]

class Server:
    """The easy-llama FastAPI server, providing a WebUI and an API router"""

    def __init__(
        self,
        thread: 'ez.Thread',
        host: str = "127.0.0.1",
        port: int = 8080
    ):
        assert_type(thread, getattr(ez, 'Thread'), 'thread', 'Server.__init__')
        self._thread = thread
        self._host = host
        self._port = port
        self._app = FastAPI(title=f"[easy-llama.Server @ {host}:{port}]")
        self._router = APIRouter(prefix="/api")

        # add CORS middleware for WebUI compatibility
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        #
        # API endpoints
        #

        router = self._router # the decorators look ugly without this

        @router.post("/send", response_model=MessageResponseModel)
        async def send(message: MessageModel = Body(...)) -> dict[str, Union[str, int]]:
            """Send a message in this thread as the user and return the generated response.
            This adds your message and the bot's message to the thread."""
            self._thread.messages.append({
                'role': message.role,
                'content': message.content
            })
            input_toks = self._thread.get_input_ids(role='bot')
            response_toks = self._thread.llama.generate(
                input_tokens=input_toks,
                n_predict=-1,
                stop_tokens=self._thread._stop_tokens,
                sampler_preset=self._thread.sampler_preset
            )
            response_txt = self._thread.llama.detokenize(response_toks, special=False)
            self._thread.messages.append({
                'role': 'bot',
                'content': response_txt
            })
            return {
                'role': 'bot',
                'content': response_txt,
                'n_input_tokens': len(input_toks),
                'n_output_tokens': len(response_toks)
            }

        @router.post("/add_message", response_model=StatusResponseModel)
        async def add_message(message: MessageModel = Body(...)) -> dict:
            """Add a message to the Thread without triggering a response"""
            self._thread.add_message(message.role, message.content)
            return {'success': True}
        
        @router.post("/set_system_prompt", response_model=StatusResponseModel)
        async def set_system_prompt(request: SetSysPromptRequestModel = Body(...)) -> dict:
            """Set the system prompt on-the-fly"""
            if len(self._thread.messages) > 0:
                try:
                    role = self._thread.messages[0]['role']
                except (IndexError, KeyError) as exc:
                    self.log('failed to set system prompt (could not get role)')
                    return {'success': True}
            

        @router.post("/trigger", response_model=MessageResponseModel)
        async def trigger() -> dict[str, Union[str, int]]:
            """Trigger a new message to be generated"""
            input_toks = self._thread.get_input_ids(role='bot')
            response_toks = self._thread.llama.generate(
                input_tokens=input_toks,
                n_predict=-1,
                stop_tokens=self._thread._stop_tokens,
                sampler_preset=self._thread.sampler_preset
            )
            response_txt = self._thread.llama.detokenize(response_toks, special=False)
            self._thread.messages.append({
                'role': 'bot',
                'content': response_txt
            })
            return {
                'role': 'bot',
                'content': response_txt,
                'n_input_tokens': len(input_toks),
                'n_output_tokens': len(response_toks)
            }

        @router.get("/messages", response_model=list[MessageModel])
        async def messages() -> list[dict]:
            """Get a list of all messages in this thread"""
            return self._thread.messages

        @router.get("/summarize", response_model=SummaryResponseModel)
        async def summarize() -> dict[str, str]:
            """Generate and return a summary of the thread content"""
            return {"summary": self._thread.summarize()}

        @router.post("/cancel")
        async def cancel() -> dict:
            """If the model is currently generating, cancel it. Otherwise, do nothing."""
            # TODO
            return {"status": "not implemented"}

        @router.post("/reset")
        async def reset() -> dict:
            """Reset the thread to its default state"""
            self._thread.reset()
            return {"status": "success"}
        
        @router.get("/info", response_model=InfoResponseModel)
        async def get_info() -> dict[str, Union[str, int, float]]:
            """Return some info about the llama model and the thread"""
            with ez.utils.suppress_output():
                input_ids = self._thread.get_input_ids(role=None)
            n_thread_tokens = len(input_ids)
            n_ctx = self._thread.llama.n_ctx()
            return {
                'model_name': self._thread.llama.name(),
                'model_n_params': self._thread.llama.n_params(),
                'model_bpw': self._thread.llama.bpw(),
                'n_tokens': self._thread.llama.pos,
                'n_ctx': n_ctx,
                'n_ctx_train': self._thread.llama.n_ctx_train(),
                'n_thread_tokens': n_thread_tokens,
                'n_thread_messages': len(self._thread.messages)
            }
        
        @router.post("/sampler", response_model=SamplerSettingsModel)
        async def sampler(settings: SamplerSettingsModel = Body(...)) -> dict[
            str, Union[int, float, list[str], dict[int, float]]
        ]:
            """Control the sampler settings over the API"""

            seed                  = settings.seed
            top_k                 = settings.top_k
            top_p                 = settings.top_p
            min_p                 = settings.min_p
            xtc_probability       = settings.xtc_probability
            xtc_threshold         = settings.xtc_threshold
            typical_p             = settings.typical_p
            temp                  = settings.temp
            dynatemp_delta        = settings.dynatemp_delta
            dynatemp_exponent     = settings.dynatemp_exponent
            penalty_last_n        = settings.penalty_last_n
            penalty_repeat        = settings.penalty_repeat
            penalty_freq          = settings.penalty_freq
            penalty_present       = settings.penalty_present
            dry_multiplier        = settings.dry_multiplier
            dry_base              = settings.dry_base
            dry_allowed_length    = settings.dry_allowed_length
            dry_penalty_last_n    = settings.dry_penalty_last_n
            mirostat              = settings.mirostat
            top_n_sigma           = settings.top_n_sigma
            mirostat_tau          = settings.mirostat_tau
            mirostat_eta          = settings.mirostat_eta
            dry_sequence_breakers = settings.dry_sequence_breakers
            logit_bias            = settings.logit_bias
            
            current = self._thread.sampler_preset
            
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
            self._thread.sampler_preset = new_preset
            
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
    
    def log(self, text: str, level: Literal[1,2,3,4] = 1) -> None:
        ez.utils.log(f'[easy-llama.Server @ {self.host}:{self.port}] {text}', level=level)

    def start(self):
        self.log('starting uvicorn')
        try:
            uvicorn.run(self.app, host=self.host, port=self.port)
        except Exception as exc:
            self.log(f'exception in uvicorn: {type(exc).__name__}: {exc}', 3)
            raise exc
        except KeyboardInterrupt:
            pass
        
        self.log(f'goodbye :)')
