# server.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""The easy-llama FastAPI server, including an API endpoint and a simple WebUI"""

# NOTE: This module is WIP.

import os
import uvicorn

import easy_llama as ez

from pydantic                import BaseModel
from fastapi.staticfiles     import StaticFiles
from fastapi.responses       import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from easy_llama.utils        import assert_type, log, ANSI
from typing                  import Optional, Union, Literal
from fastapi                 import FastAPI, APIRouter, Body, HTTPException


WEBUI_DIRECTORY = os.path.join(os.path.dirname(__file__), 'webui')

STATUS_RESPONSE_SUCCESS = {'success': True}
STATUS_RESPONSE_FAILURE = {'success': False}

Y = ANSI.FG_BRIGHT_YELLOW
R = ANSI.MODE_RESET_ALL

#
# Pydantic models for FastAPI
#

class StatusResponseModel(BaseModel):
    success: bool

class MessageModel(BaseModel):
    role:    str
    content: str

class SendMessageModel(BaseModel):
    role:      str
    content:   str
    n_predict: Optional[int] = -1

class MessageResponseModel(BaseModel):
    role:            str
    content:         str
    n_input_tokens:  int
    n_output_tokens: int

class SetSystemPromptRequestModel(BaseModel):
    content: str

class SummaryResponseModel(BaseModel):
    summary: str

class InfoResponseModel(BaseModel):
    model_name:        str
    model_n_params:    int
    model_bpw:         float
    n_tokens:          int
    n_ctx:             int
    n_ctx_train:       int
    n_thread_tokens:   int
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

    def log(self, text: str, level: Literal[1,2,3,4] = 1) -> None:
        log(f'ez.Server @ {Y}{self._host}{R}:{Y}{self._port}{R} {text}', level=level)

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
        self._app = FastAPI(title=f"ez.Server @ {host}:{port}")
        self._router = APIRouter(prefix="/api")

        self._setup_api_routes()

        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._app.include_router(self._router)

        if os.path.isdir(WEBUI_DIRECTORY):
             self._app.mount("/static", StaticFiles(directory=WEBUI_DIRECTORY), name="static")

             @self._app.get("/", include_in_schema=False)
             async def read_index():
                 index_path = os.path.join(WEBUI_DIRECTORY, 'index.html')
                 if os.path.exists(index_path):
                     return FileResponse(index_path)
                 else:
                     # fallback if index.html is missing but directory exists
                     self.log('index.html not found', 3)
                     raise HTTPException(status_code=404, detail="index.html not found")

        else:
             self.log(f"WebUI directory not found at {WEBUI_DIRECTORY}", level=2)
             @self._app.get("/", include_in_schema=False)
             async def root_fallback():
                 # provide a message if WebUI isn't found
                 return {"message": "easy-llama server is running. WebUI not found."}


    def _setup_api_routes(self):
        """Define all API endpoints."""
        router = self._router

        @router.get("/ping", response_model=StatusResponseModel)
        async def ping() -> dict:
            """Check if the server is running."""
            self.log('pong')
            return STATUS_RESPONSE_SUCCESS

        @router.post("/send", response_model=MessageResponseModel)
        async def send(message: SendMessageModel = Body(...)) -> dict[str, Union[str, int]]:
            """Send a user message and get the bot's response.
            Adds both messages to the thread history."""

            try:
                 self._thread.add_message(message.role, message.content)
            
            except ValueError as exc:
                 self.log(f"/send: error adding user message: {exc}", 3)
                 raise HTTPException(status_code=400, detail=str(exc))

            input_toks = self._thread.get_input_ids(role='bot')

            try:
                 response_toks = self._thread.llama.generate(
                     input_tokens=input_toks,
                     n_predict=message.n_predict if message.n_predict is not None else -1,
                     stop_tokens=self._thread._stop_tokens,
                     sampler_preset=self._thread.sampler_preset
                 )
                 response_txt = self._thread.llama.detokenize(response_toks, special=False)
            
            except ez.llama.ExceededContextLengthException as exc:
                 self.log(f"ExceededContextLengthException: {exc}", 3)
                 raise HTTPException(status_code=413, detail=str(exc))
            
            except Exception as exc:
                self.log(f"Error during generation: {type(exc).__name__}: {exc}", level=3)
                raise HTTPException(status_code=500, detail=str(exc))

            # Add bot response to history
            self._thread.add_message('bot', response_txt)

            return {
                'role': 'bot',
                'content': response_txt,
                'n_input_tokens': len(input_toks), # Tokens processed for this turn
                'n_output_tokens': len(response_toks) # Tokens generated this turn
            }

        @router.post("/add_message", response_model=StatusResponseModel, tags=["Chat"])
        async def add_message(message: MessageModel = Body(...)) -> dict:
            """
            Add a message to the thread history without triggering a bot response.
            Useful for manually setting up conversation state.
            """
            self.log(f"/add_message request: role='{message.role}', content='{message.content[:50]}...'")
            try:
                self._thread.add_message(message.role, message.content)
                return STATUS_RESPONSE_SUCCESS
            except ValueError as exc:
                self.log(f'Failed to add message: {exc}', 3)
                raise HTTPException(status_code=400, detail=str(exc))

        @router.post("/set_system_prompt", response_model=StatusResponseModel, tags=["Chat"])
        async def set_system_prompt(request: SetSystemPromptRequestModel = Body(...)) -> dict:
            """
            Set or update the system prompt (the first message if it has a system role).
            If no system prompt exists, it will be prepended.
            """
            content = request.content
            new_sys_msg = {'role': 'system', 'content': content}
            messages = self._thread.messages

            if messages and messages[0]['role'].lower() in self._thread.valid_system_roles:
                self.log("Updating existing system prompt.")
                messages[0]['content'] = content
            else:
                self.log("Prepending new system prompt.")
                messages.insert(0, new_sys_msg)

            orig_messages = self._thread._orig_messages
            if orig_messages and orig_messages[0]['role'].lower() in self._thread.valid_system_roles:
                 orig_messages[0]['content'] = content
            else:
                 orig_messages.insert(0, new_sys_msg)

            return STATUS_RESPONSE_SUCCESS

        @router.post("/trigger", response_model=MessageResponseModel, tags=["Chat"])
        async def trigger() -> dict[str, Union[str, int]]:
            """Trigger the bot to generate a response based on the current history,
            without adding a user message first. Appends the bot's response."""
            self.log("/trigger request received")
            if not self._thread.messages:
                 # TODO: this is incorrect
                 self.log("Cannot trigger response: No messages in history.", level=2)
                 raise HTTPException(status_code=400, detail="Cannot trigger response with empty history")

            last_message_role = self._thread.messages[-1]['role'].lower()
            if last_message_role in self._thread.valid_bot_roles:
                 self.log("Last message was from bot, triggering another bot response.", level=1)
                 # Allow triggering even if last was bot, might be desired sometimes
            elif last_message_role in self._thread.valid_system_roles:
                 self.log("Last message was system, triggering bot response.", level=1)
            elif last_message_role in self._thread.valid_user_roles:
                 self.log("Last message was user, triggering bot response.", level=1)

            # prepare for generation

            try:
                input_toks = self._thread.get_input_ids(role='bot')
            except ez.llama.ExceededContextLengthException as exc:
                 self.log(f"Context length exceeded before generation: {exc}", level=3)
                 raise HTTPException(status_code=413, detail=f"Input context too long: {exc}")
            except ValueError as exc: # Handle potential errors in get_input_ids
                 self.log(f"Error getting input IDs: {exc}", level=3)
                 raise HTTPException(status_code=500, detail=f"Internal error preparing generation: {exc}")

            # generate response

            try:
                 response_toks = self._thread.llama.generate(
                     input_tokens=input_toks,
                     n_predict=-1,
                     stop_tokens=self._thread._stop_tokens,
                     sampler_preset=self._thread.sampler_preset
                 )
                 response_txt = self._thread.llama.detokenize(response_toks, special=False).strip()
                 self.log(f"Generated {len(response_toks)} tokens: '{response_txt[:50]}...'")
            except ez.llama.ExceededContextLengthException as exc:
                 self.log(f"Context length exceeded during generation: {exc}", level=3)
                 raise HTTPException(status_code=413, detail=f"Context length exceeded during generation: {exc}")
            except Exception as exc:
                self.log(f"Error during generation: {type(exc).__name__}: {exc}", level=3)
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

            # Add bot response to history
            self._thread.add_message('bot', response_txt)

            return {
                'role': 'bot',
                'content': response_txt,
                'n_input_tokens': len(input_toks),
                'n_output_tokens': len(response_toks)
            }


        @router.get("/messages", response_model=list[MessageModel], tags=["Chat"])
        async def messages() -> list[dict]:
            """Get the current list of messages in the thread history."""
            self.log("/messages request received")
            return self._thread.messages


        @router.get("/summarize", response_model=SummaryResponseModel, tags=["Chat"])
        async def summarize() -> dict[str, str]:
            """
            Generate a summary of the current chat thread.
            Note: This performs inference and modifies the Llama context temporarily.
            """
            self.log("/summarize request received")
            try:
                summary_text = self._thread.summarize()
                self.log(f"Generated summary: '{summary_text[:50]}...'")
                return {"summary": summary_text}
            except Exception as exc:
                 self.log(f"Error during summarization: {type(exc).__name__}: {exc}", level=3)
                 raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")


        @router.post("/cancel", response_model=StatusResponseModel, tags=["Control"])
        async def cancel() -> dict:
            """(Not Implemented) Attempt to cancel ongoing generation."""
            # NOTE: llama.cpp doesn't have a built-in robust way to interrupt
            # llama_decode mid-computation from Python easily without potential
            # instability. This requires more complex handling (e.g., abort callbacks).
            self.log('Endpoint `/api/cancel` is not implemented yet!', 2)
            # return STATUS_RESPONSE_FAILURE
            raise HTTPException(status_code=501, detail="Cancel operation not implemented")


        @router.post("/reset", response_model=StatusResponseModel, tags=["Control"])
        async def reset() -> dict:
            """Reset the chat thread to its initial state (usually just the system prompt)."""
            self.log("/reset request received")
            try:
                self._thread.reset()
                self.log("Thread and Llama context reset successfully.")
                return STATUS_RESPONSE_SUCCESS
            except Exception as exc:
                self.log(f"Error during reset: {type(exc).__name__}: {exc}", level=3)
                raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


        @router.get("/info", response_model=InfoResponseModel, tags=["Status"])
        async def get_info() -> dict[str, Union[str, int, float]]:
            """Get information about the loaded model and current thread state."""
            self.log("/info request received")
            # Use suppress_output if get_input_ids might log verbosely
            # with ez.utils.suppress_output(disable=ez.get_verbose()):
            input_ids = self._thread.get_input_ids(role=None) # Get tokens just for counting

            return {
                'model_name': self._thread.llama.name(),
                'model_n_params': self._thread.llama.n_params(),
                'model_bpw': self._thread.llama.bpw(),
                'n_tokens': self._thread.llama.pos,
                'n_ctx': self._thread.llama.n_ctx(),
                'n_ctx_train': self._thread.llama.n_ctx_train(),
                'n_thread_tokens': len(input_ids),
                'n_thread_messages': len(self._thread.messages)
            }

        @router.post("/sampler", response_model=SamplerSettingsModel, tags=["Sampling"])
        async def set_sampler(settings: SamplerSettingsModel = Body(...)) -> dict:
            """
            Update the sampler settings for subsequent generations.
            Returns the complete current settings after applying the update.
            """
            self.log(f"/sampler POST request received with updates: {settings.model_dump(exclude_unset=True)}")
            current = self._thread.sampler_preset
            update_data = settings.model_dump(exclude_unset=True) # Get only provided fields

            # Create a dictionary from the current preset
            current_data = current.as_dict()

            # Update the current data with the new values
            current_data.update(update_data)

            # Create a new preset from the merged data
            try:
                new_preset = ez.SamplerPreset(**current_data)
                # Update the thread's active sampler preset
                self._thread.sampler_preset = new_preset
                self.log(f"Sampler settings updated. Current: {new_preset}")
                return new_preset.as_dict() # Return the full new settings
            except Exception as exc: # Catch potential validation errors in SamplerPreset
                self.log(f"Error updating sampler settings: {exc}", level=3)
                raise HTTPException(status_code=400, detail=f"Invalid sampler settings: {exc}")


        @router.get("/sampler", response_model=SamplerSettingsModel, tags=["Sampling"])
        async def get_sampler() -> dict:
            """Get the current sampler settings."""
            self.log("/sampler GET request received")
            return self._thread.sampler_preset.as_dict()


    # --- Server Start Method ---
    def start(self):
        """Start the Uvicorn server."""
        try:
            uvicorn.run(
                 app=self._app,
                 host=self._host,
                 port=self._port
             )
        except KeyboardInterrupt:
            self.log('KeyboardInterrupt')
        except Exception as exc:
            self.log(f'Server crashed: {type(exc).__name__}: {exc}', 3)
            # Potentially re-raise or handle specific exceptions (e.g., port in use)
            raise exc
        finally:
             self.log("Server shutdown complete.")
