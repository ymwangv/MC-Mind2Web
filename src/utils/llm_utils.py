import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "gpt": {
        "default": "gpt-4.1-mini",
        "models": {
            "gpt-5": {"input_cost": 1.25, "cached_input_cost": 0.125, "output_cost": 10.00},
            "gpt-4.1": {"input_cost": 2.00, "cached_input_cost": 0.50, "output_cost": 8.00},
            "gpt-4o": {"input_cost": 2.50, "cached_input_cost": 1.25, "output_cost": 10.00},
            "gpt-5-mini": {"input_cost": 0.25, "cached_input_cost": 0.025, "output_cost": 2.00},
            "gpt-4.1-mini": {"input_cost": 0.40, "cached_input_cost": 0.10, "output_cost": 1.60},
            "gpt-4o-mini": {"input_cost": 0.15, "cached_input_cost": 0.075, "output_cost": 0.60},
        },
        "env_key": "OPENAI_API_KEY"
    },
    "gemini": {
        "default": "gemini-2.5-pro",
        "models": {
            "gemini-2.5-pro": {"input_cost": 1.25, "cached_input_cost": 0.31, "output_cost": 10.00},
            "gemini-2.5-flash": {"input_cost": 0.30, "cached_input_cost": 0.075, "output_cost": 2.50},
        },
        "env_key": "GEMINI_API_KEY"
    }
}


class LLMAPIClient(ABC):
    def __init__(self, provider: str, model: str = None, api_key: str = None):
        self.provider = provider.lower()
        if self.provider not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}. Use: {list(MODEL_CONFIGS.keys())}")

        self.config = MODEL_CONFIGS[self.provider]
        self.model = model or self.config["default"]

        if self.model not in self.config["models"]:
            logger.warning(f"Model {self.model} not in config, using default: {self.config['default']}")
            self.model = self.config["default"]

        self.api_key = api_key or os.getenv(self.config["env_key"])
        if not self.api_key:
            raise ValueError(f"API key not found. Set {self.config['env_key']} environment variable or pass api_key parameter")

        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    def call_text_only(self, user_prompt: str, system_prompt: str = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        pass

    @abstractmethod
    def call_text_image(self, user_prompt: str, system_prompt: str = None, image_path: Union[str, Path] = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        pass

    def _create_base_result(self, image_path: Union[str, Path] = None) -> Dict[str, Any]:
        result = {
            "success": False,
            "response": "",
            "error": "",
            "provider": self.provider,
            "model": self.model,
            "time_seconds": 0.0
        }
        if image_path:
            result["image_path"] = str(image_path)
        return result

    def _calculate_costs(self, tokens: Dict[str, int]) -> Dict[str, float]:
        model_config = self.config["models"].get(self.model, {"input_cost": 0, "cached_input_cost": 0, "output_cost": 0})

        input_tokens = tokens.get('input_tokens', 0)
        cached_tokens = tokens.get('cached_input_tokens', 0)
        output_tokens = tokens.get('output_tokens', 0)
        regular_input_tokens = input_tokens - cached_tokens

        regular_input_cost = regular_input_tokens * model_config["input_cost"] / 1_000_000
        cached_input_cost = cached_tokens * model_config.get("cached_input_cost", model_config["input_cost"]) / 1_000_000
        output_cost = output_tokens * model_config["output_cost"] / 1_000_000
        total_cost = regular_input_cost + cached_input_cost + output_cost

        return {
            "regular_input_cost": round(regular_input_cost, 8),
            "cached_input_cost": round(cached_input_cost, 8),
            "output_cost": round(output_cost, 8),
            "total_cost": round(total_cost, 8)
        }


class GPTClient(LLMAPIClient):
    def _initialize_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image_path: Union[str, Path]) -> Optional[str]:
        try:
            image_path = Path(image_path)
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def _extract_tokens(self, response) -> Dict[str, int]:
        tokens = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0
        }

        if hasattr(response, 'usage'):
            tokens["input_tokens"] = response.usage.input_tokens
            tokens["output_tokens"] = response.usage.output_tokens
            tokens["total_tokens"] = response.usage.total_tokens

            if hasattr(response.usage, 'input_tokens_details'):
                tokens["cached_input_tokens"] = getattr(response.usage.input_tokens_details, 'cached_tokens', 0)
            if hasattr(response.usage, 'output_tokens_details'):
                tokens["reasoning_tokens"] = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0)

        return tokens

    def call_text_only(self, user_prompt: str, system_prompt: str = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        result = self._create_base_result()
        start_time = time.time()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            if "gpt-5" in self.model:
                reasoning_budget = 1024
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    max_output_tokens=max_tokens + reasoning_budget
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )

            result["response"] = response.output_text.strip()
            result["success"] = True

            tokens = self._extract_tokens(response)
            result["tokens"] = tokens
            result["costs"] = self._calculate_costs(tokens)

        except Exception as e:
            result["error"] = f"OpenAI API error: {e}"
            logger.error(f"OpenAI API error: {e}")

        result["time_seconds"] = time.time() - start_time
        return result

    def call_text_image(self, user_prompt: str, system_prompt: str = None, image_path: Union[str, Path] = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        result = self._create_base_result(image_path)
        start_time = time.time()

        base64_image = self._encode_image(image_path)
        if not base64_image:
            result["error"] = "Failed to encode image"
            return result

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            })

            if "gpt-5" in self.model:
                reasoning_budget = 1024
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    max_output_tokens=max_tokens + reasoning_budget
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )

            result["response"] = response.output_text.strip()
            result["success"] = True

            tokens = self._extract_tokens(response)
            result["tokens"] = tokens
            result["costs"] = self._calculate_costs(tokens)

        except Exception as e:
            result["error"] = f"OpenAI API error: {e}"
            logger.error(f"OpenAI API error: {e}")

        result["time_seconds"] = time.time() - start_time
        return result


class GeminiClient(LLMAPIClient):
    def _initialize_client(self):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI library not found. Install with: pip install google-genai")
        self.client = genai.Client(api_key=self.api_key)

    def _extract_tokens(self, response) -> Dict[str, int]:
        tokens = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0
        }

        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            candidates_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            thoughts_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

            tokens["input_tokens"] = getattr(usage, 'prompt_token_count', 0) or 0
            tokens["cached_input_tokens"] = getattr(usage, 'cached_content_token_count', 0) or 0
            tokens["output_tokens"] = candidates_tokens + thoughts_tokens
            tokens["reasoning_tokens"] = thoughts_tokens
            tokens["total_tokens"] = getattr(usage, 'total_token_count', 0) or 0

        return tokens

    def call_text_only(self, user_prompt: str, system_prompt: str = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        result = self._create_base_result()
        start_time = time.time()

        try:
            thinking_budget = 1024
            config_params = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
                "max_output_tokens": max_tokens + thinking_budget
            }
            if system_prompt:
                config_params["system_instruction"] = system_prompt

            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_params),
                contents=[user_prompt]
            )

            if response and response.text:
                result["response"] = response.text.strip()
                result["success"] = True

                tokens = self._extract_tokens(response)
                result["tokens"] = tokens
                result["costs"] = self._calculate_costs(tokens)
            else:
                result["error"] = "Empty response from Gemini"

        except Exception as e:
            result["error"] = f"Gemini API error: {e}"
            logger.error(f"Gemini API error: {e}")

        result["time_seconds"] = time.time() - start_time
        return result

    def call_text_image(self, user_prompt: str, system_prompt: str = None, image_path: Union[str, Path] = None, temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
        result = self._create_base_result(image_path)
        start_time = time.time()

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            thinking_budget = 1024
            config_params = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
                "max_output_tokens": max_tokens + thinking_budget
            }
            if system_prompt:
                config_params["system_instruction"] = system_prompt

            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(**config_params),
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    user_prompt
                ]
            )

            if response and response.text:
                result["response"] = response.text.strip()
                result["success"] = True

                tokens = self._extract_tokens(response)
                result["tokens"] = tokens
                result["costs"] = self._calculate_costs(tokens)
            else:
                result["error"] = "Empty response from Gemini"

        except Exception as e:
            result["error"] = f"Gemini API error: {e}"
            logger.error(f"Gemini API error: {e}")

        result["time_seconds"] = time.time() - start_time
        return result


def create_llm_client(provider: str = "gpt", model: str = None, api_key: str = None) -> LLMAPIClient:
    provider = provider.lower()
    if provider == "gpt":
        return GPTClient(provider, model, api_key)
    elif provider == "gemini":
        return GeminiClient(provider, model, api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use: gpt, gemini")


def call_llms_text_only(user_prompt: str, system_prompt: str = None, provider: str = "gpt", model: str = None, api_key: str = None,
                       temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
    client = create_llm_client(provider=provider, model=model, api_key=api_key)
    return client.call_text_only(user_prompt, system_prompt, temperature, max_tokens)


def call_llms_text_image(user_prompt: str, system_prompt: str = None, image_path: Union[str, Path] = None,
                        provider: str = "gpt", model: str = None, api_key: str = None,
                        temperature: float = 1.0, max_tokens: int = 1024) -> Dict[str, Any]:
    client = create_llm_client(provider=provider, model=model, api_key=api_key)
    return client.call_text_image(user_prompt, system_prompt, image_path, temperature, max_tokens)

