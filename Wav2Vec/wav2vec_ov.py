import torch
import torch.nn as nn
import fairseq
import librosa
import torch.nn.functional as F
import torchaudio.functional as F_audio
import os
import time
import openvino as ov
from typing import Tuple


torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])


class Wav2VecEncoderScriptable(nn.Module):
    """
    Fully TorchScript-compatible encoder wrapper.
    This version is designed to be traced with torch.jit.trace.
    """
    def __init__(self, w2v_encoder):
        super().__init__()
        self.encoder = w2v_encoder

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Forward without padding_mask for simpler tracing
        encoder_out = self.encoder(source, None)
        if isinstance(encoder_out, dict):
            logits = encoder_out['encoder_out']
            predicted_ids = torch.argmax(logits, axis=-1)
            predicted_ids = self.remove_consecutive_duplicates(predicted_ids.T, dim=1)
            return predicted_ids
        return encoder_out
    
    def remove_consecutive_duplicates(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Drop in replacement for torch.unique_consecutive that is fully TorchScript compatible."""
        """The issue is unsupported dynamic operators."""
        # x shape: (B, T)
        
        # Compare with shifted version
        shifted = torch.roll(x, shifts=1, dims=dim)
        
        # First element must be kept
        mask = x != shifted
        mask[:, 0] = True
        
        return x[mask].view(x.size(0), -1)


class AudioToText:
    DEFAULT_SAMPLING_RATE = int(os.getenv("DEFAULT_SAMPLING_RATE", "16000"))
    PAD_OFFSET = int(os.getenv("ASR_PAD_OFFSET", "200000"))

    def __init__(
        self,
        model_path: str = "hindi.pt",
        warmup_iterations: int = 1,
        sample_audio_path: str = "./samples/hindi.wav",
        batch_size: int = 5,
        dtype: str = "float32",
        use_ov: bool = True,
        device: str = "GPU"
    ):
        self.batch_size = batch_size
        self.use_ov = use_ov

        # Load the fairseq model
        model, self.cfg, self.task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.base_model = model[0]
        self.dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        self.base_model.to(self.dtype)
        self.base_model.eval()

        # Create TorchScript-compatible wrapper
        self.encoder_wrapper = Wav2VecEncoderScriptable(self.base_model.w2v_encoder)
        self.encoder_wrapper.eval()

        if use_ov:
            print("Converting model to OV format...")
            # Create example input for tracing
            example_input = torch.zeros([1, 160000], dtype=self.dtype)
            
            self.model = ov.convert_model(self.encoder_wrapper, example_input=example_input)
            self.model = ov.compile_model(self.model, device_name=device)
            print("OV conversion complete!")
        else:
            self.model = self.encoder_wrapper

        self.token = self.task.target_dictionary
        self.warmup_audio_path = sample_audio_path
        self.warmup(warmup_iterations)

    def warmup(self, warmup_iters: int):
        for _ in range(warmup_iters):
            _ = self.transcribe(self.warmup_audio_path)

    def preprocess_audio(self, path: str, sample_rate: int = 16000) -> torch.Tensor:
        """Preprocess audio file for model inference."""
        audio, sr = librosa.load(path, sr=sample_rate)
        waveform = torch.tensor(audio)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed
        if sr != sample_rate:
            waveform = F_audio.resample(
                waveform,
                orig_freq=sr,
                new_freq=sample_rate
            )

        # Normalize (equivalent to sox norm)
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        waveform = waveform.to(self.dtype)

        # Apply layer normalization
        waveform = F.layer_norm(waveform, waveform.shape)

        return waveform

    def decode_predictions(self, logits: torch.Tensor) -> list:
        """Decode model logits to transcriptions."""
        predicted_ids = logits[0]
        transcriptions = []
        for ids in predicted_ids:
            transcription = self.token.string(ids)
            transcription = transcription.replace(' ', "").replace('|', " ").strip()
            transcriptions.append(transcription)

        return transcriptions

    def transcribe(self, path: str, sample_rate: int = 16000) -> Tuple[list, float]:
        """Transcribe audio file to text."""
        input_sample = self.preprocess_audio(path, sample_rate)
        
        st = time.perf_counter()

        with torch.no_grad():
            logits = self.model(input_sample)

        transcriptions = self.decode_predictions(logits)
        total_time = time.perf_counter() - st

        return transcriptions, total_time

    def audio_to_text(self, audio_path: str) -> Tuple[str, float]:
        """Transcribe a single audio file."""
        transcriptions, time_taken = self.transcribe(audio_path)
        return transcriptions[0] if transcriptions else "", time_taken

    def save_ov_model(self, save_path: str):
        """Save the OV model to disk."""
        if self.use_ov:
            ov.save_model(self.model, save_path)
            print(f"TorchScript model saved to: {save_path}")
        else:
            print("Model is not in OV format.")

    @staticmethod
    def load_torchscript_model(model_path: str) -> torch.jit.ScriptModule:
        """Load a saved TorchScript model."""
        return torch.jit.load(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio to Text Transcription (TorchScript)")
    parser.add_argument("--model_path", type=str, default="hindi.pt", 
                        help="Path to the ASR model file")
    parser.add_argument("--sample_audio_path", type=str, default="./samples/hindi.wav", 
                        help="Path to a sample audio file for warmup")
    parser.add_argument("--audio_path", type=str, default="./samples/hindi2.wav", 
                        help="Path to the audio file to transcribe")
    parser.add_argument("--warmup_iterations", type=int, default=3, 
                        help="Number of warmup iterations")
    parser.add_argument("--dtype", type=str, default="float32", 
                        help="Data type for model inference (e.g., float32, bfloat16)")
    parser.add_argument("--save_torchscript", type=str, default=None,
                        help="Path to save the TorchScript model (optional)")
    parser.add_argument("--no_ov", action="store_true",
                        help="Disable TorchScript conversion (use eager mode)")
    parser.add_argument("--device", type=str, default="GPU",
                        help="Device for OV inference (e.g., CPU, GPU)")
    args = parser.parse_args()

    audio_to_text = AudioToText(
        model_path=args.model_path,
        warmup_iterations=args.warmup_iterations,
        sample_audio_path=args.sample_audio_path,
        dtype=args.dtype,
        use_ov=not args.no_ov,
        device=args.device
    )

    # Optionally save the TorchScript model
    if args.save_torchscript:
        audio_to_text.save_torchscript_model(args.save_torchscript)

    # Run transcription
    transcription, total_time = audio_to_text.transcribe(args.audio_path)
    print("Transcription:", transcription)
    print("Total Time:", total_time)

    transcription, total_time = audio_to_text.audio_to_text("samples/hindi.wav")
    print("Transcription:", transcription)
    print("Total Time:", total_time)
