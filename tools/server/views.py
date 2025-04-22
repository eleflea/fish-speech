import io
import os
import time
from http import HTTPStatus

import numpy as np
import ormsgpack
import soundfile as sf
from pydub import AudioSegment
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    request,
)
from loguru import logger
from typing_extensions import Annotated

from fish_speech.utils.schema import (
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    model_manager.ensure_inference_engine()
    decoder_model = model_manager.decoder_model

    # Encode the audio
    start_time = time.time()
    tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
    logger.info(f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms")

    # Return the response
    return ormsgpack.packb(
        ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    model_manager.ensure_inference_engine()
    decoder_model = model_manager.decoder_model

    # Decode the audio
    tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
    start_time = time.time()
    audios = batch_vqgan_decode(decoder_model, tokens)
    logger.info(f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms")
    audios = [audio.astype(np.float16).tobytes() for audio in audios]

    # Return the response
    return ormsgpack.packb(
        ServeVQGANDecodeResponse(audios=audios),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    # Get the model from the app
    app_state = request.app.state
    model_manager: ModelManager = app_state.model_manager
    model_manager.ensure_inference_engine()
    engine = model_manager.tts_inference_engine
    sample_rate = engine.decoder_model.sample_rate

    # Check if the text is too long
    if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content=f"Text is too long, max length is {app_state.max_text_length}",
        )

    # Check if streaming is enabled
    if req.streaming and req.format != "wav":
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content="Streaming only supports WAV format",
        )

    # Perform TTS
    if req.streaming:
        return StreamResponse(
            iterable=inference_async(req, engine),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    else:
        fake_audios = next(inference(req, engine))
        buffer = io.BytesIO()
        if req.format == "m4a":
            if isinstance(fake_audios, np.ndarray):
                fake_audios = (fake_audios * 32767).astype(np.int16).tobytes()
            audio_segment = AudioSegment(
                fake_audios,
                frame_rate=sample_rate,
                sample_width=2,  # int16 -> 2 bytes
                channels=1,
            )
            ffmpeg_params = [
                "-c:a", "libfdk_aac",
                "-b:a", "32k",
                "-movflags", "+faststart",
            ]
            audio_segment.export(buffer, format="ipod", parameters=ffmpeg_params)  # "ipod" 是 m4a (AAC) 的别名
        else:
            sf.write(
                buffer,
                fake_audios,
                sample_rate,
                format=req.format,
            )

        return StreamResponse(
            iterable=buffer_to_async_generator(buffer.getvalue()),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
