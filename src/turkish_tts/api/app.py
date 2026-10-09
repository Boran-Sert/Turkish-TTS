"""FastAPI application wiring the model pool to the HTTP and WebSocket routers."""

import asyncio
import os
from contextlib import aclosing, asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from ..engine import pool, synthesize_stream
from ..logger import setup_logger
from ..settings import settings
from . import routes_http, routes_ws

logger = setup_logger()

WARMUP_TEXT = "Merhaba, bu bir ısındırma cümlesidir."

_INDEX_HTML = open(
    os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8"
).read()


async def _warmup() -> None:
    """Pushes one synthesis through every pooled model in the background."""
    try:
        for _ in range(pool.size):
            async with pool.acquire() as model:
                async with aclosing(synthesize_stream(model, WARMUP_TEXT)) as chunks:
                    async for _chunk in chunks:
                        pass
        logger.info("Isındırma tamamlandı, sistem tam hızda.")
    except Exception as exc:
        logger.exception(f"Isındırma başarısız: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model pool before serving, then warms it up in the background."""
    logger.info("TTS motorları başlatılıyor...")
    await pool.initialize()
    app.state.warmup_task = asyncio.create_task(_warmup())
    yield
    app.state.warmup_task.cancel()
    logger.info("TTS motorları kapatılıyor.")


app = FastAPI(title="Turkish TTS", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=settings.api.cors_allowed_methods,
    allow_headers=settings.api.cors_allowed_headers,
)

app.include_router(routes_http.router)
app.include_router(routes_ws.router)


@app.get("/", response_class=HTMLResponse)
async def serve_web_ui() -> HTMLResponse:
    """Serves the bundled demo page."""
    return HTMLResponse(content=_INDEX_HTML)


def main() -> None:
    """Runs the server, honouring TTS_HOST and TTS_PORT."""
    uvicorn.run(
        "turkish_tts.api.app:app",
        host=os.getenv("TTS_HOST", "0.0.0.0"),
        port=int(os.getenv("TTS_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
