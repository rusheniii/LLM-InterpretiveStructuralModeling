from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch

from tenacity import RetryError

import llmism.LLMClient as llm_module
from llmism.LLMClient import LLMClient, append_message


class TestLLMClient(IsolatedAsyncioTestCase):
    def test_get_client_builds_async_openai_client(self) -> None:
        expected_client = Mock(name="async_openai")
        expected_http_client = Mock(name="http_client")

        with (
            patch.object(llm_module.httpx, "AsyncClient") as async_client,
            patch.object(llm_module, "AsyncOpenAI") as async_openai,
        ):
            async_client.return_value = expected_http_client
            async_openai.return_value = expected_client

            result = llm_module._get_client("https://example.test", "api-key")

        self.assertIs(result, expected_client)
        async_client.assert_called_once_with(verify=True)
        async_openai.assert_called_once_with(
            api_key="api-key",
            base_url="https://example.test",
            http_client=expected_http_client,
        )

    def test_get_client_reraises_client_construction_error(self) -> None:
        expected_error = RuntimeError("client failed")

        with (
            patch.object(llm_module.httpx, "AsyncClient") as async_client,
            patch.object(llm_module, "AsyncOpenAI", side_effect=expected_error),
        ):
            with self.assertRaisesRegex(RuntimeError, "client failed"):
                llm_module._get_client("https://example.test", "api-key")

        async_client.assert_called_once_with(verify=True)

    def test_append_message_appends_role_and_content(self) -> None:
        expected = [{"role": "user", "content": "hello"}]
        messages = []

        append_message("user", "hello", messages)

        self.assertEqual(messages, expected)

    def test_llm_client_initializes_underlying_client_and_models(self) -> None:
        expected_client = Mock()
        result = LLMClient(expected_client, "model", "embedding-model")

        self.assertIs(result.client, expected_client)
        self.assertEqual(result.model, "model")
        self.assertEqual(result.embedding_model, "embedding-model")
        self.assertTrue(result.responses_api_supported)

    async def test_get_embedding_returns_first_embedding(self) -> None:
        expected = [0.1, 0.2]
        client = LLMClient.__new__(LLMClient)
        client.embedding_model = "embedding-model"
        client.client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[SimpleNamespace(embedding=expected)]
                    )
                )
            )
        )

        result = await client.get_embedding("text")

        self.assertEqual(result, expected)
        client.client.embeddings.create.assert_awaited_once_with(
            model="embedding-model",
            input="text",
        )

    async def test_completions_parse_returns_parsed_message(self) -> None:
        expected = {"parsed": True}
        client = LLMClient.__new__(LLMClient)
        client.model = "chat-model"
        parse = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=expected))]
            )
        )
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )

        result = await client.completions_parse(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            reasoning_effort="low",
            response_format=dict,
        )

        self.assertEqual(result, expected)
        parse.assert_awaited_once_with(
            model="chat-model",
            messages=[
                {"role": "developer", "content": "instructions"},
                {"role": "user", "content": "hi"},
            ],
            temperature=0.2,
            reasoning_effort="low",
            response_format=dict,
        )

    async def test_responses_parse_returns_output_parsed(self) -> None:
        expected = {"parsed": True}
        client = LLMClient.__new__(LLMClient)
        client.model = "response-model"
        parse = AsyncMock(return_value=SimpleNamespace(output_parsed=expected))
        client.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

        result = await client.responses_parse(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            response_format=dict,
        )

        self.assertEqual(result, expected)
        parse.assert_awaited_once_with(
            instructions="instructions",
            model="response-model",
            input=[{"role": "user", "content": "hi"}],
            text_format=dict,
            temperature=0.2,
            reasoning={"effort": "high"},
        )

    async def test_completions_parse_retries_three_times_before_failing(self) -> None:
        expected_error = RuntimeError("completions boom")
        client = LLMClient.__new__(LLMClient)
        client.model = "chat-model"
        parse = AsyncMock(side_effect=expected_error)
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )

        with self.assertRaises(RetryError):
            await client.completions_parse(
                instructions="instructions",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(parse.await_count, 3)

    async def test_responses_parse_retries_three_times_before_failing(self) -> None:
        expected_error = RuntimeError("responses boom")
        client = LLMClient.__new__(LLMClient)
        client.model = "response-model"
        parse = AsyncMock(side_effect=expected_error)
        client.client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

        with self.assertRaises(RetryError):
            await client.responses_parse(
                instructions="instructions",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(parse.await_count, 3)

    async def test_inference_uses_responses_api_when_supported(self) -> None:
        expected = {"parsed": "responses"}
        client = LLMClient.__new__(LLMClient)
        client.responses_api_supported = True
        client.responses_parse = AsyncMock(return_value=expected)
        client.completions_parse = AsyncMock()

        result = await client.inference(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="medium",
            response_format=dict,
        )

        self.assertEqual(result, expected)
        client.responses_parse.assert_awaited_once_with(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            reasoning_effort="medium",
            response_format=dict,
        )
        client.completions_parse.assert_not_awaited()

    async def test_inference_uses_completions_api_when_not_supported(self) -> None:
        expected = {"parsed": "completions"}
        client = LLMClient.__new__(LLMClient)
        client.responses_api_supported = False
        client.responses_parse = AsyncMock()
        client.completions_parse = AsyncMock(return_value=expected)

        result = await client.inference(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="low",
            response_format=dict,
        )

        self.assertEqual(result, expected)
        client.completions_parse.assert_awaited_once_with(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            reasoning_effort="low",
            response_format=dict,
        )
        client.responses_parse.assert_not_awaited()

    async def test_inference_drops_temperature_when_reasoning_effort_set(self) -> None:
        expected = {"parsed": "responses"}
        client = LLMClient.__new__(LLMClient)
        client.responses_api_supported = True
        client.responses_parse = AsyncMock(return_value=expected)

        with self.assertLogs(llm_module.logger, level="WARNING") as logs:
            result = await client.inference(
                instructions="instructions",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                reasoning_effort="high",
                response_format=dict,
            )

        self.assertEqual(result, expected)
        self.assertIn(
            "Temperature cannot be used when reasoning effort is specified.",
            logs.output[0],
        )
        client.responses_parse.assert_awaited_once_with(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            reasoning_effort="high",
            response_format=dict,
        )

    async def test_inference_keeps_temperature_when_reasoning_effort_empty(self) -> None:
        expected = {"parsed": "completions"}
        client = LLMClient.__new__(LLMClient)
        client.responses_api_supported = False
        client.completions_parse = AsyncMock(return_value=expected)

        result = await client.inference(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            reasoning_effort="",
            response_format=dict,
        )

        self.assertEqual(result, expected)
        client.completions_parse.assert_awaited_once_with(
            instructions="instructions",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            reasoning_effort="",
            response_format=dict,
        )
