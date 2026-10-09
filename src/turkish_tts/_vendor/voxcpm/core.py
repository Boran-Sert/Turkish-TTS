# NOTICE: This file has been modified by the project contributors.
# Modified to comply with Apache License 2.0, Section 4(b).

import os
import sys
import re
import json
import tempfile
import time
import numpy as np
import logging
from typing import Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig
from .model.voxcpm2 import VoxCPM2Model
from .model.utils import next_and_close

import logging
logger = logging.getLogger("TTS_SYSTEM")
from .config_loader import config_instance, StreamingConfigModel
from .streaming import AudioChunk, AudioFormatConverter
from .text_buffer import StreamingTextSource

class VoxCPM:
    def __init__(
        self,
        voxcpm_model_path: str,
        zipenhancer_model_path: str | None = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
        optimize: bool = True,
        device: str | None = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
        max_length: Optional[int] = None,
    ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            device: Runtime device. If set to ``None`` or ``"auto"``, VoxCPM
                will choose automatically (preferring CUDA, then MPS, then CPU).
                If set explicitly, that device is used or a clear error is raised.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded.
        """
        print(
            f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}",
            file=sys.stderr,
        )

        # If lora_weights_path is provided but no lora_config, load the saved
        # lora_config.json (so r/alpha match the checkpoint); else use a default.
        if lora_weights_path is not None and lora_config is None:
            cfg_path = os.path.join(lora_weights_path, "lora_config.json")
            if os.path.isdir(lora_weights_path) and os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    lora_config = LoRAConfig(**json.load(f)["lora_config"])
                print(f"Loaded LoRAConfig from: {cfg_path}", file=sys.stderr)
            else:
                lora_config = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False)
                print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}", file=sys.stderr)

        # Determine model type from config.json architecture field
        config_path = os.path.join(voxcpm_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        arch = config.get("architecture", "voxcpm").lower()

        if arch == "voxcpm2":
            self.tts_model = VoxCPM2Model.from_local(
                voxcpm_model_path,
                optimize=optimize,
                device=device,
                lora_config=lora_config,
                max_length=max_length,
            )
            print("Loaded VoxCPM2Model", file=sys.stderr)
        elif arch == "voxcpm":
            self.tts_model = VoxCPMModel.from_local(
                voxcpm_model_path,
                optimize=optimize,
                device=device,
                lora_config=lora_config,
            )
            print("Loaded VoxCPMModel", file=sys.stderr)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}", file=sys.stderr)
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}", file=sys.stderr)

        self.text_normalizer = None
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer

            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM2",
        load_denoiser: bool = True,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str = None,
        local_files_only: bool = False,
        optimize: bool = True,
        device: str | None = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
        **kwargs,
    ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            device: Runtime device. Use ``None``/``"auto"`` for automatic
                fallback, or an explicit value such as ``"cpu"``, ``"mps"``,
                ``"cuda"``, or ``"cuda:0"``.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is
                provided without lora_config, a default config will be created with
                enable_lm=True and enable_dit=True.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded
                after model initialization.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def generate_stream(
        self,
        text: str,
        *,
        streaming_config: Optional[StreamingConfigModel] = None,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: Optional[int] = None,
    ) -> Generator[AudioChunk, None, None]:
        """Yields audio chunks at specified durations during the generation process."""
        cfg = streaming_config or config_instance.streaming
        inf_steps = inference_timesteps or config_instance.model.inference_timesteps
        cfg_val = cfg_value or config_instance.model.cfg_value
        sample_rate = self.tts_model.sample_rate
        
        target_samples = int((cfg.chunk_duration_ms / 1000.0) * sample_rate)
        
        generator = self._generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            reference_wav_path=reference_wav_path,
            cfg_value=cfg_val,
            inference_timesteps=inf_steps,
            min_len=min_len,
            max_len=max_len,
            normalize=normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            streaming=True,
            seed=seed
        )
        
        buffer_list = []
        current_samples = 0
        chunk_idx = 0
        
        # --- Crossfade (Overlap-Add) Setup ---
        # VoxCPM2's StreamingVAEDecoder carries causal-conv padding state
        # between decode_chunk() calls, producing seamless inter-chunk audio.
        # Manual crossfade is unnecessary for v2 and harmful: it discards
        # 480 samples (10 ms @ 48 kHz) from every patch, causing audible
        # data loss — especially on short utterances where total audio is
        # only a few thousand samples.
        is_v2_model = isinstance(self.tts_model, VoxCPM2Model)
        crossfade_ms = 0.0 if is_v2_model else 10.0
        crossfade_samples = int((crossfade_ms / 1000.0) * sample_rate)
        if crossfade_samples > 0:
            window = np.hanning(crossfade_samples * 2)
            fade_in = window[:crossfade_samples]
            fade_out = window[crossfade_samples:]
        else:
            fade_in, fade_out = np.array([]), np.array([])
            
        previous_overlap = None
        
        try:
            for patch_np in generator:
                if patch_np.ndim > 1:
                    patch_np = patch_np.squeeze()
                    
                # 1. Apply Crossfade (Overlap-Add)
                if previous_overlap is not None and crossfade_samples > 0:
                    cf_len = min(len(previous_overlap), len(patch_np), crossfade_samples)
                    if cf_len > 0:
                        patch_np[:cf_len] = (previous_overlap[-cf_len:] * fade_out[-cf_len:]) + (patch_np[:cf_len] * fade_in[:cf_len])
                        
                # 2. Keep the end of this patch for the NEXT overlap
                if crossfade_samples > 0 and len(patch_np) > crossfade_samples:
                    previous_overlap = patch_np[-crossfade_samples:].copy()
                    patch_to_yield = patch_np[:-crossfade_samples]
                else:
                    previous_overlap = None
                    patch_to_yield = patch_np
                    
                if len(patch_to_yield) == 0:
                    continue

                buffer_list.append(patch_to_yield)
                current_samples += len(patch_to_yield)
                
                if current_samples >= target_samples:
                    # O(N) concat & dönüştürme (for-loops yok)
                    combined = np.concatenate(buffer_list, axis=-1)
                    converted = AudioFormatConverter.convert(combined, sample_rate, cfg.output_format)
                    
                    yield AudioChunk(
                        data=converted,
                        sample_rate=sample_rate,
                        chunk_index=chunk_idx,
                        sentence_index=0, 
                        timestamp_ms=time.time() * 1000,
                        is_final=False,
                        is_sentence_final=False
                    )
                    
                    chunk_idx += 1
                    buffer_list = []
                    current_samples = 0
                    
            if previous_overlap is not None:
                buffer_list.append(previous_overlap)
                
            if buffer_list:
                combined = np.concatenate(buffer_list, axis=-1)
                converted = AudioFormatConverter.convert(combined, sample_rate, cfg.output_format)
                yield AudioChunk(
                    data=converted,
                    sample_rate=sample_rate,
                    chunk_index=chunk_idx,
                    sentence_index=0,
                    timestamp_ms=time.time() * 1000,
                    is_final=True,
                    is_sentence_final=True
                )
            else:
                yield AudioChunk(
                    data=b"" if "pcm" in cfg.output_format or "float" in cfg.output_format else np.array([]),
                    sample_rate=sample_rate,
                    chunk_index=chunk_idx,
                    sentence_index=0,
                    timestamp_ms=time.time() * 1000,
                    is_final=True,
                    is_sentence_final=True
                )
                
        finally:
            generator.close()

    def generate_stream_from_text_source(
        self,
        text_source: StreamingTextSource,
        *,
        streaming_config: Optional[StreamingConfigModel] = None,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: Optional[float] = None,
        inference_timesteps: Optional[int] = None,
    ) -> Generator[AudioChunk, None, None]:
        """Synthesizes text sequentially from a continuous stream of incoming tokens."""
        cfg = streaming_config or config_instance.streaming
        
        curr_prompt_wav = prompt_wav_path
        curr_prompt_text = prompt_text
        temp_files = []
        sentence_idx = 0
        
        sample_rate = self.tts_model.sample_rate
        
        # Pydantic v1 ve v2 uyumluluğu için model_dump() veya dict()
        cfg_dict = getattr(cfg, "model_dump", getattr(cfg, "dict"))()
        
        try:
            for sentence in text_source:
                inf_steps = inference_timesteps or config_instance.model.inference_timesteps
                if config_instance.model.adaptive_timesteps:
                    if len(sentence) < 30:
                        inf_steps = config_instance.model.adaptive_timesteps_short
                    elif len(sentence) > 100:
                        inf_steps = config_instance.model.adaptive_timesteps_long

                actual_prompt_wav = curr_prompt_wav if cfg.enable_lookbehind else None
                actual_prompt_txt = curr_prompt_text if cfg.enable_lookbehind else None

                # Lookbehind caching gerektiriyorsa formatı geçici olarak numpy'a ayarlıyoruz.
                override_cfg_dict = cfg_dict.copy()
                if cfg.enable_lookbehind:
                    override_cfg_dict["output_format"] = "numpy"
                override_cfg = type(cfg)(**override_cfg_dict)

                logger.debug(f"[Audio Generation] Sentez başlatıldı: '{sentence}' (Timesteps: {inf_steps})")
                start_time = time.time()

                gen = self.generate_stream(
                    text=sentence,
                    streaming_config=override_cfg,
                    prompt_wav_path=actual_prompt_wav,
                    prompt_text=actual_prompt_txt,
                    reference_wav_path=reference_wav_path,
                    cfg_value=cfg_value,
                    inference_timesteps=inf_steps
                )
                
                full_sentence_audio_np = []
                
                for chunk in gen:
                    chunk_start_time = time.time()
                    chunk.sentence_index = sentence_idx
                    chunk.is_sentence_final = chunk.is_final
                    chunk.is_final = False # Sadece en sonda True olacak
                    
                    if cfg.enable_lookbehind:
                        full_sentence_audio_np.append(chunk.data)
                        if cfg.output_format != "numpy":
                            chunk.data = AudioFormatConverter.convert(chunk.data, sample_rate, cfg.output_format)

                    chunk_size = len(chunk.data) if hasattr(chunk.data, "__len__") else 0
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"[Audio Chunk] {chunk_size} bytes üretildi. (Cümle: {sentence_idx}, Format: {cfg.output_format}, Süre: {elapsed_ms:.1f}ms)")
                    
                    yield chunk

                if cfg.enable_lookbehind and full_sentence_audio_np:
                    mode = getattr(cfg, "lookbehind_mode", "strict")
                    if mode == "anchor":
                        # Anchor modunda, prompt sadece bir kez belirlenir.
                        # Eğer kullanıcı baştan bir prompt vermediyse, İLK üretilen cümle
                        # referans alınır ve tüm metin boyunca sabit tutulur.
                        # Bu sayede sesin ortalara doğru bozulması %100 engellenir.
                        if curr_prompt_wav is None:
                            combined_np = np.concatenate(full_sentence_audio_np, axis=-1)
                            import scipy.io.wavfile
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                            pcm_16 = (np.clip(combined_np, -1.0, 1.0) * 32767.0).astype('<i2')
                            scipy.io.wavfile.write(tmp.name, sample_rate, pcm_16)
                            temp_files.append(tmp.name)
                            
                            curr_prompt_wav = tmp.name
                            curr_prompt_text = sentence
                    else:
                        # strict mod (zincirleme): Her cümlenin sesi, bir sonraki cümleye 
                        # referans olarak aktarılır (degradation'a yol açabilir).
                        combined_np = np.concatenate(full_sentence_audio_np, axis=-1)
                        import scipy.io.wavfile
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        pcm_16 = (np.clip(combined_np, -1.0, 1.0) * 32767.0).astype('<i2')
                        scipy.io.wavfile.write(tmp.name, sample_rate, pcm_16)
                        temp_files.append(tmp.name)
                        
                        curr_prompt_wav = tmp.name
                        curr_prompt_text = sentence
                
                sentence_idx += 1
                logger.debug(f"[Audio Generation] Cümle {sentence_idx-1} tamamlandı. Toplam süre: {(time.time() - start_time) * 1000:.1f}ms")
                
            yield AudioChunk(
                data=b"" if "pcm" in cfg.output_format or "float" in cfg.output_format else np.array([]),
                sample_rate=sample_rate,
                chunk_index=0,
                sentence_index=sentence_idx,
                timestamp_ms=time.time() * 1000,
                is_final=True,
                is_sentence_final=True
            )

        finally:
            for tmp_path in temp_files:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def _generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        seed: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        Args:
            text: Input text to synthesize.
            prompt_wav_path: Path to prompt audio for continuation mode.
                Must be paired with ``prompt_text``.
            prompt_text: Text content corresponding to the prompt audio.
            reference_wav_path: Path to reference audio for voice cloning
                (structurally isolated via ref_audio tokens). Can be used
                alone or combined with ``prompt_wav_path`` + ``prompt_text``.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            min_len: Minimum audio length.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt/reference audio if a
                denoiser is available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
            seed: Optional random seed for reproducibility.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU.
            Yields audio chunks for each generation step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if reference_wav_path is not None:
            if not os.path.exists(reference_wav_path):
                raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        is_v2 = isinstance(self.tts_model, VoxCPM2Model)
        if reference_wav_path is not None and not is_v2:
            raise ValueError("reference_wav_path is only supported with VoxCPM2 models")

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        temp_files = []

        try:
            actual_prompt_path = prompt_wav_path
            actual_ref_path = reference_wav_path

            if denoise and self.denoiser is not None:
                if prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_files[-1])
                    actual_prompt_path = temp_files[-1]
                if reference_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temp_files[-1])
                    actual_ref_path = temp_files[-1]

            if actual_prompt_path is not None or actual_ref_path is not None:
                if is_v2:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                        reference_wav_path=actual_ref_path,
                    )
                else:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                    )
            else:
                fixed_prompt_cache = None

            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer

                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
                seed=seed,
            )

            if streaming:
                try:
                    for wav, _, _ in generate_result:
                        float_audio = wav.squeeze(0).cpu().numpy()
                        if np.isnan(float_audio).any() or np.max(np.abs(float_audio)) == 0.0:
                            logger.error("DİKKAT: Model NaN veya tamamen sessiz bir dalga üretti! Timesteps yetersiz.")
                        yield float_audio
                finally:
                    generate_result.close()
            else:
                wav, _, _ = next_and_close(generate_result)
                float_audio = wav.squeeze(0).cpu().numpy()
                if np.isnan(float_audio).any() or np.max(np.abs(float_audio)) == 0.0:
                    logger.error("DİKKAT: Model NaN veya tamamen sessiz bir dalga üretti! Timesteps yetersiz.")
                yield float_audio

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VoxCPMModel)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file.

        Args:
            lora_weights_path: Path to LoRA weights (.pth file or directory
                containing lora_weights.ckpt).

        Returns:
            tuple: (loaded_keys, skipped_keys) - lists of loaded and skipped parameter names.

        Raises:
            RuntimeError: If model was not initialized with LoRA config.
        """
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state (effectively disabling LoRA)."""
        self.tts_model.reset_lora_weights()

    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers without unloading weights.

        Args:
            enabled: If True, LoRA layers are active; if False, only base model is used.
        """
        self.tts_model.set_lora_enabled(enabled)

    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters state dict.

        Returns:
            dict: State dict containing all LoRA parameters (lora_A, lora_B).
        """
        return self.tts_model.get_lora_state_dict()

    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None
