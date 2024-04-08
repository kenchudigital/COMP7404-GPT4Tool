"""Wrapper around HuggingFace APIs."""
import torch
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer

DEFAULT_REPO_ID = "gpt2"
VALID_TASKS = ("text2text-generation", "text-generation")


class LlamaHuggingFace:

    def __init__(self, 
                 base_model,
                 lora_model,
                 task='text-generation',
                 device='cpu',
                 max_new_tokens=512, #512
                 temperature=0.1,
                 top_p=0.75, 
                 top_k=40, # 40
                 num_beams=1,
                 cache_dir=None):
        self.task = task
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            cache_dir=cache_dir,
            padding_side="right",
            use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        print(f"Loading {base_model}...")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            cache_dir=cache_dir,
            torch_dtype=torch.float16, #float16
            low_cpu_mem_usage=True
            )
        
        print(f"Loading LoRA weights from {lora_model}")
        model = PeftModel.from_pretrained(model, lora_model)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print(0)
        self.model = model
        self.model.to(device)
        print(1)
        print('device ', device)
        if device == "cpu":
            self.model.to(torch.float16)
        else:
            self.model.to(torch.float16)
        print(2) 
        self.model.eval()
        print(3)

    @torch.no_grad()
    def __call__(self, inputs, params):
        if inputs.endswith('Thought:'):
            inputs = inputs[:-len('Thought:')]
        inputs = inputs.replace('Observation:\n\nObservation:', 'Observation:')
        inputs = "USER: \n" + inputs + 'ASSISTANT:'
        input_ids = self.tokenizer(inputs, return_tensors="pt").to(self.device).input_ids

        generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams)

        generate_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            max_new_tokens=self.max_new_tokens)
        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        response = [res.replace('ASSISTANT:', '') for res in response]
        response = [{'generated_text': res} for res in response]
        return response


class LlamaLangChain(LLM, BaseModel):
    """Wrapper around LLAMA  models.
    """

    client: Any  #: :meta private:
    repo_id: str = DEFAULT_REPO_ID
    """Model name to use."""
    task: Optional[str] = "text-generation" 
    """Task to call the model with. Should be a task that returns `generated_text`."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def set_llm_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.client, key, value)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        repo_id = values["repo_id"]
        model_kwargs = values.get("model_kwargs")
        client = LlamaHuggingFace(
            base_model=model_kwargs.get("base_model"),
            lora_model=model_kwargs.get("lora_model"),
            task=values.get("task"),
            device=model_kwargs.get("device"),
            max_new_tokens=model_kwargs.get("max_new_tokens"),
            temperature=model_kwargs.get("temperature"),
            top_p=model_kwargs.get("top_p"),
            top_k=model_kwargs.get("top_k"),
            num_beams=model_kwargs.get("num_beams"),
            cache_dir=model_kwargs.get("cache_dir"),
        )
        if client.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {client.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        values["client"] = client
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"repo_id": self.repo_id, "task": self.task},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface_hub"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to HuggingFace Hub's inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = hf("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        response = self.client(inputs=prompt, params=_model_kwargs)
        if "error" in response:
            raise ValueError(f"Error raised by inference API: {response['error']}")
        if self.client.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt) :]
        elif self.client.task == "text2text-generation":
            text = response[0]["generated_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.client.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
