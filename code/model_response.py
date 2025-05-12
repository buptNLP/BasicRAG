# from __future__ import annotations
import json
import os
import re
from typing import Any, ClassVar, Generator
import requests
from pydantic import PrivateAttr
import torch
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


class ApiModel:
    '''
    调用模型API，本示例中以使用阿里云平台调用qwen2.5-14b-instruct为例。
    可根据自己实际情况修改。
    '''
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.model_name = "qwen2.5-14b-instruct"

    def chat(self, prompt: str, **generation_params: Any) -> str:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": self.model_name,
            "input": {"prompt": prompt},
            "parameters": generation_params
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # 自动处理HTTP错误
        return response.json()['output']['text'].strip()


def remove_special_tokens(text: str) -> str:
    """清理模型生成结果中的特殊标记"""
    return re.sub(r'\[gMASK]|\bsop\b', '', text).strip()


class LocalPeftModel:
    '''
    加载本地模型
    '''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置设备编号

    def __init__(self, peft_model_path: str = "weight/lora_2", use_base_model: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "../../Model/Qwen2.5-1.5B-Ins", # 模型本地路径
            trust_remote_code=True
        )
        # 加载原始模型
        if use_base_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                "../../Model/Qwen2.5-1.5B-Ins",  
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        # 加载微调后的模型
        else:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                peft_model_path,
                trust_remote_code=True,
                device_map="auto"
            ).eval()

    def chat(self, prompt: str) -> str:
        """本地模型推理方法"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        return remove_special_tokens(response)


class MyLocalLLM(CustomLLM):
    '''
    封装与本地模型的交互
    '''
    context_window: ClassVar[int] = 2048
    num_output: ClassVar[int] = 256
    model_name: ClassVar[str] = "Qwen2.5-1.5B-Ins"

    _model: LocalPeftModel = PrivateAttr()

    def __init__(self, model_path: str):
        super().__init__()
        self._model = LocalPeftModel(model_path)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._model.chat(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        response = self._model.chat(prompt)
        for token in response.split():
            yield CompletionResponse(text=token, delta=token)


class MyApiLLM(CustomLLM):
    '''
    封装与调用模型的交互
    '''
    context_window: ClassVar[int] = 8192
    num_output: ClassVar[int] = 256
    model_name: ClassVar[str] = "qwen2.5-14b-instruct"

    _model: ApiModel = PrivateAttr()

    def __init__(self, api_key: str):
        super().__init__()
        self._model = ApiModel(api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._model.chat(prompt, **kwargs))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        response = self._model.chat(prompt, **kwargs)
        for token in response.split():
            yield CompletionResponse(text=token, delta=token)


if __name__ == '__main__':
    # 使用示例
    llm = MyApiLLM(api_key="你的API密钥")
    # llm = MyLocalLLM(model_path="../../Model/Qwen2.5-1.5B-Ins")
    print(llm.complete("林俊杰").text)