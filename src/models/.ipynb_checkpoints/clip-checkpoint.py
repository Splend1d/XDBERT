import hashlib
import os
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .clip_model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:        
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    return list(_MODELS.keys())


def load_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True, lin = False):
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    model_path = _download(_MODELS[name])
    model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
    n_px = model.input_resolution.item()

    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    #print("md")
    if not jit:
        print("not jit")
        model = build_model(model.state_dict(),lin)#.to(device)
        
        return model, transform

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if device == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, transform


def tokenize_clip(texts: Union[str, List[str]], context_length: int = 77, assign_seg = None):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    #print(texts)

    all_tokens = [_tokenizer.encode(text) for text in texts]
    

    max_seg = max([len(t)//(context_length - 2) for t in all_tokens]) if not assign_seg else assign_seg # deleted + 1 for no pad
    assert context_length <= 77
    result = torch.zeros(len(all_tokens),max_seg, 77, dtype=torch.long)
    #visn_input_mask = torch.ones(len(all_tokens),max_seg, dtype=torch.long)
    for i in range(len(all_tokens)):
        for nseg in range(min(len(all_tokens[i])//(context_length - 2)+1,max_seg)):
            start = nseg * (context_length - 2)
            end = min((nseg+1) * (context_length - 2),len(all_tokens[i]))
            if end <= start:
                break
            #if len(tokens) > context_length:
                #raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
                #force cut at 77
            #tokens = all_tokens[i][0:context_length - 1] + [tokens[i][-1]]
            #else:
            #print(i,nseg)
            result[i, nseg,:end-start+2] = torch.tensor([sot_token] + all_tokens[i][start:end] + [eot_token])
        #for nmask in range(len(all_tokens[i])//(context_length - 2)+1,max_seg):
            #visn_input_mask[i,nmask] = 0

    # result shape : bs, seg, 77
    # where seg * 77 ~= 512
    #print(result.shape,result)
    #print(visn_input_mask.shape,visn_input_mask)
    #s()
    return result

def tokenize_clip_multi(texts: Union[str, List[str]], context_length: int = 77):
    result = torch.zeros(len(texts),4, 77, dtype=torch.long)
    #bs,12,77
    max_seg = 0
    for n,pair in enumerate(texts):
        #print(pair)
        tokens = None
        for t in pair:
            if tokens is None:
                tokens = tokenize_clip(t,assign_seg = 2)
            else:
                tokens = torch.cat((tokens,tokenize_clip(t,assign_seg = 2)),dim = 1)
            #print(t)
            #print(tokens,tokens.shape)

        result[n,:tokens.shape[1],:] = tokens
        #s()
        max_seg = max(max_seg,tokens.shape[1])
        #print(tokens.shape)
    #print(max_seg)
    #result = result[:,:max_seg,:]
    #print(result.shape)
    #print(result)
    #s()
    return result
