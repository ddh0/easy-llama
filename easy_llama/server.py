# server.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""The easy-llama FastAPI server, including an API endpoint and a WebUI"""

import os
import sys
import time
import uuid
import uvicorn

import easy_llama as ez

from fastapi             import FastAPI, APIRouter, HTTPException, status
from typing              import List, Optional, Literal
from pydantic            import BaseModel, Field
from easy_llama.utils    import assert_type, log
from fastapi.staticfiles import StaticFiles

class Server:
    """The easy-llama FastAPI server, providing a WebUI and an API endpoint"""

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

        # Add CORS middleware for WebUI compatibility
        self.add_cors()

        # Set up API router
        self.api_router = APIRouter(prefix="/api")
        self.setup_api_endpoints()

        # Mount components
        self.app.include_router(self.api_router)
        self.app.mount(
            "/",
            StaticFiles(
                directory="webui",
                html=True
            ),
            name=f"[easy-llama.Server (WebUI) @ {host}:{port}]"
        )

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
        async def send(content: str) -> str:
            """Send a message and receive the response as a string"""
            response = self.thread.send(content)
            return response

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
            seed: Optional[int] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            min_p: Optional[float] = None,
            temp: Optional[float] = None
        ) -> None:
            """Control the most common sampler settings over the API"""
            self.thread.sampler_preset = ez.SamplerPreset(
                seed=seed if seed is not None else self.thread.sampler_preset.seed,
                top_k=top_k if top_k is not None else self.thread.sampler_preset.top_k,
                top_p=top_p if top_p is not None else self.thread.sampler_preset.top_p,
                min_p=min_p if min_p is not None else self.thread.sampler_preset.min_p,
                temp=temp if temp is not None else self.thread.sampler_preset.temp
            )

        @self.api_router.get("/summarize")
        async def summarize() -> str:
            """Generate and return a summary of the thread content"""
            return self.thread.summarize()

        @self.api_router.get("/info")
        async def get_info() -> dict:
            """Return some info about the llama model and the context usage"""
            return {
                'model': self.thread.llama.name(),
                'n_tokens_used': len(self.thread.get_input_ids(role=None)),
                'n_ctx': self.thread.llama.n_ctx(),
                'n_ctx_train': self.thread.llama.n_ctx_train(),
                'n_messages': len(self.thread.messages),
            }

        @self.api_router.post("/cancel")
        async def cancel() -> None:
            """If the model is currently generating, cancel it. Otherwise, do nothing."""
            # TODO: Implement cancellation logic if needed
            pass

        @self.api_router.post("/reset")
        async def reset() -> None:
            """Reset the thread to its default state"""
            self.thread.reset()

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
