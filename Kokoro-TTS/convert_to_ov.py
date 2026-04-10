import torch
import openvino as ov
from pathlib import Path
from kokoro import KPipeline
from huggingface_hub import hf_hub_download

import gc


def get_model_dir(model_id):
    return Path(model_id.split("/")[-1])


def convert_model(model_id: str):

    pipeline = KPipeline(lang_code="a", repo_id=model_id)

    model = pipeline.model
    model.forward = model.forward_with_tokens
    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,), dtype=torch.float32)

    model_dir = get_model_dir(model_id)
    ov_model = ov.convert_model(model, example_input=(input_ids, style, speed), input=[ov.PartialShape("[1, 2..]"), ov.PartialShape([1, -1])])
    ov.save_model(ov_model, model_dir / "openvino_model.xml")
    hf_hub_download(repo_id=model_id, filename="config.json", local_dir=model_dir)
    print("Model successfully saved at :", model_dir)
    del model 
    del pipeline

    gc.collect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Model ID on huggingface", default="hexgrad/Kokoro-82M")

    args = parser.parse_args()

    convert_model(args.model_id)    



