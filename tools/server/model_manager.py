import threading
import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
        idle_timeout: int = 0,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path
        self.decoder_config_name = decoder_config_name
        self.idle_timeout = idle_timeout

        self.precision = torch.half if half else torch.bfloat16

        self.lock = threading.Lock()
        self.unload_timer = None

        if self.idle_timeout <= 0:
            self.load_inference_engine()
        
            # Warm up the models
            if self.mode == "tts":
                self.warm_up(self.tts_inference_engine)

    def load_inference_engine(self) -> None:
        # Load the TTS models
        self.load_llama_model(
            self.llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        self.load_decoder_model(
            self.decoder_config_name, self.decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )

    def ensure_inference_engine(self) -> None:
        if hasattr(self, "tts_inference_engine"):
            self.reset_unload_timer()
            return

        with self.lock:
            if not hasattr(self, "tts_inference_engine"):
                self.load_inference_engine()
        self.reset_unload_timer()

    def unload_inference_engine(self) -> None:
        unload_attrs = [
            "llama_queue",
            "decoder_model",
            "tts_inference_engine",
        ]
        with self.lock:
            removed_attrs = []
            for attr in unload_attrs:
                if hasattr(self, attr):
                    delattr(self, attr)
                    removed_attrs.append(attr)
            import gc
            gc.collect()
            logger.info(f"Unloaded models: {removed_attrs}")

            if self.unload_timer:
                self.unload_timer = None

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Decoder model loaded.")
    
    def reset_unload_timer(self) -> None:
        with self.lock:
            if self.unload_timer is not None:
                self.unload_timer.cancel()

            if self.idle_timeout > 0:
                self.unload_timer = threading.Timer(self.idle_timeout, self.unload_inference_engine)
                self.unload_timer.start()

    def warm_up(self, tts_inference_engine) -> None:
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        logger.info("Models warmed up.")
