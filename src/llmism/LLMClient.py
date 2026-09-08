from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt
import httpx
import logging

logger = logging.getLogger(__name__)

def _get_client(openai_base_url: str, openai_api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_base_url,
        http_client=httpx.AsyncClient(verify=True),
    )

def append_message(role, content, messages):
    messages.append({"role": role, "content": content})

class LLMClient(object):
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        embedding_model: str,
        responses_api_supported: bool = True,
    ):
        self.client = client
        self.model = model
        self.embedding_model = embedding_model
        self.responses_api_supported = responses_api_supported

    async def get_embedding(self, text: str):
        embedding = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return embedding.data[0].embedding

    @retry(stop=stop_after_attempt(3))
    async def completions_parse(
        self,
        instructions: str,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str = "high",
        response_format=None,
    ):
        completions_messages = []
        append_message("developer", instructions, completions_messages)
        completions_messages.extend(messages)
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=completions_messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        )
        return response.choices[0].message.parsed

    @retry(stop=stop_after_attempt(3))
    async def responses_parse(
        self,
        instructions: str,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str = "high",
        response_format=None,
    ):
        response = await self.client.responses.parse(
            instructions=instructions,
            model=self.model,
            input=messages,
            text_format=response_format,
            reasoning = {
                "effort": reasoning_effort
            },
            temperature=temperature,
        )
        return response.output_parsed

    async def inference(
        self,
        instructions: str,
        messages: list[dict],
        temperature: float = 0.0,
        reasoning_effort: str = None,
        response_format=None,
    ):        
        if reasoning_effort and reasoning_effort != "" and temperature is not None:
            logger.warning("Temperature cannot be used when reasoning effort is specified.")
            temperature = None
        if self.responses_api_supported:
            return await self.responses_parse(
                instructions=instructions,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )
        else:
            return await self.completions_parse(
                instructions=instructions,
                messages=messages,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            )