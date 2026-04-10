from text_preprocess_for_inference import TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
import json
import torch
import yaml
import sys
from utilities import SAMPLING_RATE, WARMUP_PARAGRAPHS
from datetime import datetime
import os
import time
import numpy as np
import openvino as ov

sys.path.append(os.getenv("HIFIGAN_PATH", f"hifigan"))
from hifigan.env import AttrDict
from hifigan.models import Generator
from hifigan.meldataset import MAX_WAV_VALUE
import torch.nn.functional as F

import nltk
nltk.download('averaged_perceptron_tagger_eng')

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEFAULT_VALUE = 600


def load_hifigan_vocoder(language: str, gender: str, device: str, dtype: str = "float32"):
    """
    Loads HiFi-GAN vocoder configuration file and generator model.
    """
    vocoder_config = f"vocoder/{gender}/{language}/config.json"
    vocoder_generator = f"vocoder/{gender}/{language}/generator"

    if not os.path.exists(vocoder_config) or not os.path.exists(vocoder_generator):
        raise FileNotFoundError(
            f"Vocoder files not found. Expected config: {vocoder_config}, generator: {vocoder_generator}")

    with open(vocoder_config, 'r') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    if dtype == "bfloat16":
        generator = generator.to(torch.bfloat16)

    return generator


def load_fastspeech2_model(language: str, gender: str, device: str, dtype: str = "float32"):
    """
    Loads FastSpeech2 model and updates its configuration with absolute paths.
    """
    config_path = f"{language}/{gender}/model/config.yaml"
    tts_model_path = f"{language}/{gender}/model/model.pth"

    if not os.path.exists(config_path) or not os.path.exists(tts_model_path):
        raise FileNotFoundError(
            f"FastSpeech2 model files not found. Expected config: {config_path}, model: {tts_model_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    current_working_directory = os.getcwd()
    feat_rel_path = "model/feats_stats.npz"
    pitch_rel_path = "model/pitch_stats.npz"
    energy_rel_path = "model/energy_stats.npz"

    feat_path = os.path.join(current_working_directory,
                             language, gender, feat_rel_path)
    pitch_path = os.path.join(
        current_working_directory, language, gender, pitch_rel_path)
    energy_path = os.path.join(
        current_working_directory, language, gender, energy_rel_path)

    config["normalize_conf"]["stats_file"] = feat_path
    config["pitch_normalize_conf"]["stats_file"] = pitch_path
    config["energy_normalize_conf"]["stats_file"] = energy_path

    # Temporarily write the modified config to a new file or use a BytesIO object if preferred
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    model = Text2Speech(train_config=config_path, model_file=tts_model_path, device=device, vocoder_config=None,vocoder_file=None)
    model.vocoder=None
    
    if dtype == "bfloat16":
        model.model = model.model.to(torch.bfloat16)

    return model


def split_into_chunks(text: str, words_per_chunk: int = 100):
    """Splits text into chunks of specified words_per_chunk."""
    words = text.split()
    chunks = [words[i:i + words_per_chunk]
              for i in range(0, len(words), words_per_chunk)]
    return [' '.join(chunk) for chunk in chunks]


class Text2SpeechApp:
    def __init__(self, language: str, batch_size: str = 1, alpha: float = 1, dtype: str = "bfloat16"):
        self.alpha = alpha
        self.lang = language
        self.batch_size = batch_size
        self.dtype = dtype
        self.vocoder_model = {}
        self.fastspeech2_model = {}
        self.supported_genders = []
        
        self.preprocessor = TTSDurAlignPreprocessor()

        genders = ["male", "female"]
        for gender in genders:
            try:
                self.vocoder_model[gender] = load_hifigan_vocoder(
                    f"{language}_latest", gender, device, self.dtype)
                with torch.no_grad():
                    self.vocoder_model[gender] = ov.convert_model(self.vocoder_model[gender], example_input=torch.ones([1, 160, MAX_DEFAULT_VALUE]))
                
                self.vocoder_model[gender] = ov.compile_model(self.vocoder_model[gender], device_name="GPU")
                print(
                    f"Loaded HiFi-GAN vocoder for {language}-{gender}")

                self.fastspeech2_model[gender] = load_fastspeech2_model(
                    f"{language}_latest", gender, device, self.dtype)
                
                # with torch.no_grad():
                    # self.fastspeech2_model[gender] = ov.convert_model(self.fastspeech2_model[gender])
                print(
                    f"Loaded FastSpeech2 model for {language}-{gender}")
                self.supported_genders.append(gender)
            except FileNotFoundError as e:
                print(
                    f"Error loading model for {language}-{gender}: {e}. This model key will not be available.")
            except Exception as e:
                print(
                    f"An unexpected error occurred while loading model for {language}-{gender}: {e}. This model key will not be available.")
        self.warmup()

    def pre_print(self, print_str: str):
        print("=================================================")
        print(print_str)
        print("=================================================")

    def warmup(self):
        self.pre_print("TTS Warming up!")

        lang = self.lang.lower()
        text = WARMUP_PARAGRAPHS.get(lang)

        if not text:
            print(f"No warmup paragraph available for language: {lang}")
            return

        # Ensure warmup output directory exists
        output_dir = "./warmup_outputs"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Running warmup for language: {lang}")
        print(f"Warmup text length: {len(text.split())} words")

        total_start_time = time.time()

        for gender in ["male", "female"]:
            if gender not in self.fastspeech2_model:
                print(f"Skipping warmup for {gender} - model not loaded.")
                continue

            print(f"Starting warmup for {lang}-{gender}")
            try:
                gender_start_time = time.time()
                for i in range(2):  # Run twice; adjust as needed
                    print(f"Warmup iteration {i + 1} for {gender}")
                    time_taken, _ = self.convert_and_save(
                        text=text,
                        speaker_gender=gender,
                        output_file_dir=output_dir
                    )
                    print(f"Iteration {i + 1} for {gender} completed in {time_taken:.2f} seconds")
                gender_total_time = time.time() - gender_start_time
                print(f"Total warmup time for {gender}: {gender_total_time:.2f} seconds")
            except Exception as e:
                print(f"Warmup failed for {lang}-{gender}: {e}")

        total_time = time.time() - total_start_time
        print(f"Total TTS warmup completed in {total_time:.2f} seconds")
        self.pre_print("TTS Warming finished!")

    def save_to_file(self, audio_arr, file_path):
        write(file_path, SAMPLING_RATE, audio_arr)
        print(f"Audio saved to {file_path}")

    def convert_and_save(self, text: str, speaker_gender="male", output_file_dir: str = "./outputs"):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_file = f"{output_file_dir}/{self.lang}_{speaker_gender}_{timestamp}.wav"

        start = time.time()
        audio_arr = []
        result_chunks = split_into_chunks(text)

        for chunk_text in result_chunks:
            # Preprocess the text
            preprocessed_text, _ = self.preprocessor.preprocess(
                chunk_text, self.lang, speaker_gender)
            preprocessed_text = " ".join(preprocessed_text)

            with torch.no_grad():
                # Generate mel-spectrograms
                out = self.fastspeech2_model[speaker_gender](preprocessed_text,
                                             decode_conf={"alpha": self.alpha})

                x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262

                # Convert mel-spectrograms to raw audio waveforms
                y_g_hat = self.vocoder_model[speaker_gender](x)
                audio = y_g_hat.squeeze()

                audio = audio * MAX_WAV_VALUE
            
            audio = audio.numpy().astype('int16')
            audio_arr.append(audio)

        result_array = np.concatenate(audio_arr, axis=0)
        self.save_to_file(audio_arr=result_array, file_path=output_file)
        time_taken = time.time() - start

        return time_taken, output_file
    

    def generate_audio_bytes(self, text: str, speaker_gender="male", save_file: bool = False):
            preprocessed_text, _ = self.preprocessor.preprocess(
                text, self.lang, speaker_gender)
            preprocessed_text = " ".join(preprocessed_text)

            with torch.no_grad():
                # Generate mel-spectrograms
                st = time.perf_counter()
                out = self.fastspeech2_model[speaker_gender](preprocessed_text,
                                             decode_conf={"alpha": self.alpha})

                x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262

                # Convert mel-spectrograms to raw audio waveforms
                # [8,8,8,2]; default sr = 44100
                trim_length = int(8 * 8 * 8 * 2 * x.shape[-1]) 
                x = F.pad(x, (0, MAX_DEFAULT_VALUE - x.shape[-1]), value=-12)
                st = time.perf_counter()
                y_g_hat = self.vocoder_model[speaker_gender](x)
                audio = y_g_hat[0][0][0][:trim_length]

                audio = audio * MAX_WAV_VALUE

            return audio
    

    def evaluate_performance(self, input_sentences: list, save_file: bool = False):
        total_sentences = len(input_sentences)
        print(f"\nTotal T2S to be done: {total_sentences}\n")
        for i, sentence in enumerate(input_sentences):
            start_time = time.perf_counter()
            audio = self.generate_audio_bytes(text=sentence)
            time_taken = time.perf_counter() - start_time
            print("=================================================")
            print(f"Sentence {i + 1}/{total_sentences}:{sentence} processed in {time_taken:.2f} seconds")

            if save_file:
                os.makedirs(f"audios_{self.dtype}/numpy_files", exist_ok=True)
                os.makedirs(f"audios_{self.dtype}/audio_files", exist_ok=True)

                output_file = f"audios_{self.dtype}/numpy_files/file_{i}.npy"

                if audio.dtype == torch.bfloat16:
                    audio = audio.to(torch.float32)
                    
                audio = audio.astype('int16')
                np.save(output_file, audio)

                audio_file_path = f"audios_{self.dtype}/audio_files/file_{i}.wav"
                with open(audio_file_path, "wb") as f:    
                    write(f, SAMPLING_RATE, audio)
                print(f"Audio saved to {audio_file_path}") 
                       
        return time_taken


    def save_to_files(self, byte_ios, file_prefix: str) -> list[str]:
        file_paths = []
        for i in range(len(byte_ios)):
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_path = f"{file_prefix}_{timestamp}_{i + 1}.wav"
            file_paths.append(file_path)
            with open(file_path, "wb") as f:
                f.write(byte_ios[i].read())
            print(f"Audio saved to {file_path}")
        return file_paths

    def batch_convert_and_save(self, input_sentences: list[str], speaker_gender="male", output_file_dir: str = "./outputs"):
        start_time = time.time()
        output_file_paths = []
        total_sentences = len(input_sentences)
        os.makedirs(output_file_dir, exist_ok=True)

        print(f"Total T2S to be done: {total_sentences}\n")
        combined_para = ''.join(input_sentences)
        paragraph_time, output_path = self.convert_and_save(
            combined_para, speaker_gender=speaker_gender, output_file_dir=output_file_dir)
        print(f"Paragraph Time: {paragraph_time}\n")
        output_file_paths.append(output_path)

        time_taken = time.time() - start_time
        return time_taken, output_file_paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text to Speech benchmarking")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for TTS inference")
    parser.add_argument("--language", type=str, default="hindi", help="Language for TTS")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for FastSpeech2 decoding")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for model inference")
    args = parser.parse_args()

    batch_size = 1
    language = "hindi"
    alpha = 1
    tts = Text2SpeechApp(batch_size=batch_size, alpha=alpha, language=language, dtype=args.dtype)
    st = time.perf_counter()
    texts = [
        "जीवन में सफलता पाने के लिए केवल सपने देखना ही नहीं, बल्कि उन्हें पूरा करने के लिए निरंतर प्रयास और आत्मविश्वास भी ज़रूरी होता है।",
        "कठिन परिस्थितियाँ हमें तोड़ने नहीं आतीं, बल्कि हमें मज़बूत बनाकर जीवन के असली अर्थ से परिचित कराती हैं।",
        "सकारात्मक सोच और सही दृष्टिकोण के साथ किया गया हर छोटा प्रयास भी एक दिन बड़ी उपलब्धि में बदल जाता है।",
        "जब हम निस्वार्थ भाव से दूसरों की मदद करते हैं, तब हमारे अपने जीवन में भी शांति और संतुलन अपने आप आ जाता है।"
    ]

    total_time = tts.evaluate_performance(texts, save_file=True)
    et = time.perf_counter()
    print(f"Total time for evaluating {len(texts)} sentences: {et - st:.2f} seconds")
    print(f"Average time per sentence: {(et - st)/len(texts):.2f} seconds")