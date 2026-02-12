import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.llm_utils import call_llms_text_only, call_llms_text_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClarificationResult:
    clarification_need: bool = False
    clarification_question: str = ""
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class ApiMeta:
    attempts: int = 0
    time_seconds: float = 0.0
    tokens: Dict[str, Any] = None
    costs: Dict[str, Any] = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = {}
        if self.costs is None:
            self.costs = {}


@dataclass
class QueryResponse:
    status: str = "failed"
    error: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    response: str = ""
    result: ClarificationResult = None
    api_meta: ApiMeta = None

    def __post_init__(self):
        if self.result is None:
            self.result = ClarificationResult()
        if self.api_meta is None:
            self.api_meta = ApiMeta()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response": self.response,
            "result": {
                "clarification_need": self.result.clarification_need,
                "clarification_question": self.result.clarification_question,
                "reasoning": self.result.reasoning,
                "confidence": self.result.confidence
            },
            "api_meta": {
                "time_seconds": self.api_meta.time_seconds,
                "attempts": self.api_meta.attempts,
                "tokens": self.api_meta.tokens,
                "costs": self.api_meta.costs
            }
        }


class ClarificationAgent:

    def __init__(self, provider: str, model: str, key_number: int, mode: str, env_type: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.key_number = key_number
        self.mode = mode
        self.env_type = env_type
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> Optional[str]:
        api_dir = Path(__file__).parent.parent.parent / "utils" / "api"
        api_file = api_dir / f"{self.provider}.json"

        if not api_file.exists():
            logger.warning(f"API configuration file not found: {api_file}")
            return None

        try:
            with open(api_file, 'r') as f:
                api_config = json.load(f)

            key_name = f"{self.provider}_{self.key_number}"

            if key_name not in api_config:
                available_keys = [k for k in api_config.keys() if not k.startswith('_')]
                logger.warning(f"API key '{key_name}' not found. Available keys: {available_keys}")
                return None

            logger.info(f"Using API key: {key_name}")
            return api_config[key_name]
        except Exception as e:
            logger.error(f"Error loading API key: {e}")
            return None

    def call_api_with_retry(self, system_prompt: str, user_prompt: str, screenshot_path: Optional[str] = None,
                            temperature: float = 0.0, max_tokens: int = 4096, max_retries: int = 3) -> Dict[str, Any]:

        total_processing_time = 0.0

        for attempt in range(max_retries):
            time.sleep(1)
            total_processing_time += 1

            if self.mode == "conv_only" or self.env_type == "html":
                result = call_llms_text_only(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    provider=self.provider,
                    model=self.model,
                    api_key=self.api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                result = call_llms_text_image(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    image_path=screenshot_path,
                    provider=self.provider,
                    model=self.model,
                    api_key=self.api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            total_processing_time += result.get('time_seconds', 0)

            if result['success']:
                result['attempts'] = attempt + 1
                result['time_seconds'] = total_processing_time
                return result
            else:
                if attempt == max_retries - 1:
                    result['attempts'] = attempt + 1
                    result['time_seconds'] = total_processing_time
                    return result

                wait_time = 2 ** attempt
                total_processing_time += wait_time
                time.sleep(wait_time)

    def parse_response(self, response_text: str) -> Dict[str, Any]:
        response_text = response_text.strip()

        if response_text.startswith('```json') and response_text.endswith('```'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```') and response_text.endswith('```'):
            response_text = response_text[3:-3].strip()

        return json.loads(response_text)

    def process_query(self, query_data: Dict[str, Any]) -> QueryResponse:

        system_prompt = query_data.get('system_prompt')
        user_prompt = query_data.get('user_prompt')
        screenshot_path = query_data.get('screenshot')

        if not system_prompt or not user_prompt:
            logger.error("Missing prompts in query data")
            return QueryResponse(
                status="failed",
                error="Missing prompts",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response="",
                result=ClarificationResult(),
                api_meta=ApiMeta()
            )

        api_result = self.call_api_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            screenshot_path=screenshot_path
        )

        if not api_result.get('success'):
            return QueryResponse(
                status="failed",
                error=api_result.get('error', 'Unknown API error'),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response="",
                result=ClarificationResult(),
                api_meta=ApiMeta(
                    attempts=api_result.get('attempts', 0),
                    time_seconds=api_result.get('time_seconds', 0.0),
                    tokens=api_result.get('tokens', {}),
                    costs=api_result.get('costs', {})
                )
            )

        response_text = api_result.get('response', '')

        try:
            response_data = self.parse_response(response_text)
            return QueryResponse(
                status="success",
                error="",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response_text,
                result=ClarificationResult(
                    clarification_need=response_data.get("clarification_need", False),
                    clarification_question=response_data.get("clarification_question", ""),
                    reasoning=response_data.get("reasoning", ""),
                    confidence=response_data.get("confidence", 0.0)
                ),
                api_meta=ApiMeta(
                    attempts=api_result.get('attempts', 0),
                    time_seconds=api_result.get('time_seconds', 0),
                    tokens=api_result.get('tokens', {}),
                    costs=api_result.get('costs', {})
                )
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return QueryResponse(
                status="failed",
                error=str(e),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response_text,
                result=ClarificationResult(),
                api_meta=ApiMeta(
                    attempts=api_result.get('attempts', 0),
                    time_seconds=api_result.get('time_seconds', 0),
                    tokens=api_result.get('tokens', {}),
                    costs=api_result.get('costs', {})
                )
            )