uv venv --python=3.12

uv pip install transformers==4.50.0 sentencepiece==0.2.1
uv pip install openvino==2025.4.1
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit
uv pip install --editable ./
cd ..

for conversion

python convert_indictrans2_ov.py  --model-name ai4bharat/indictrans2-en-indic-1B --output-dir ./openvino_models/indictrans2-en-indic-1B-fp32/ --device GPU --precision FP32

python convert_indictrans2_ov.py   --model-name ai4bharat/indictrans2-indic-indic-1B --output-dir ./openvino_models/indictrans2-indic-indic-1B-fp16/ --device GPU

OR

python convert_indictrans2_ov.py   --model-name ai4bharat/indictrans2-indic-indic-1B --output-dir ./openvino_models/indictrans2-indic-indic-1B-fp16/ --device GPU --precision FP16


for testing
python run_indictrans2_ov.py  --model-dir ./openvino_models/FP16/optimum --device GPU --src-lang hin_Deva --tgt-lang tam_Taml --warmup 10


start server

start_server_en_indic.bat
start_server_indic_indic.bat

test server

python test_nmt_en_indic.py
test_nmt_indic_indic.py


