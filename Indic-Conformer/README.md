1 pip install openvino openvino-dev numpy librosa soundfile tqdm resampy
2. put your read token in the text file hugging_face_read_token.txt
3. python download_all.py
(ignore requirements.txt that gets generated after running download_all.py).
4. python convert_to_openvino_fp32.py
5. python convert_to_openvino_fp16.py
6. python test real audio.py name of audio file hi CPU or python test real audio.py "name of audio file" hi GPU