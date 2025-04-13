# server.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

"""The easy-llama server, including an API endpoint and a WebUI"""

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
            name=f"[easy-llama WebUI @ {host}:{port}]"
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
            pass

        @self.api_router.post("/add_message")
        async def add_message(role: str, content: str) -> None:
            pass

        @self.api_router.post("/trigger")
        async def trigger() -> str:
            pass

        @self.api_router.get("/messages")
        async def messages() -> list[dict]:
            pass

        @self.api_router.post("/sampler")
        async def sampler(
            top_k: Optional[int],
            top_p: Optional[float],
            min_p: Optional[float],
            temp:  Optional[float],
        ) -> None:
            self.thread.sampler_preset
        
        @self.api_router.get("/summarize")
        async def summarize() -> str:
            return self.thread.summarize()

        @self.api_router.get("/info")
        async def get_info() -> dict:
            return {
                'model': self.thread.llama.name(),
                'n_ctx': self.thread.llama.n_ctx(),
                'n_ctx_train': self.thread.llama.n_ctx_train(),
                'n_messages': len(self.thread.messages),
                'n_tokens_used': len(self.thread.get_input_ids(role=None)),
            }

        @self.api_router.post("/reset")
        async def reset() -> None:
            self.thread.reset()

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
