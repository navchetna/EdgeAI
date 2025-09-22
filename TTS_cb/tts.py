import torchaudio as ta
import time 

from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
st = time.perf_counter()
wav = model.generate(text)
print("Time to generate: ", time.perf_counter() - st)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH="YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-2.wav", wav, model.sr)
