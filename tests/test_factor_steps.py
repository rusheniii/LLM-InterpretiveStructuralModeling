from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

from llmism.IdentifyKeyFactors import IdentifyKeyFactors
from llmism.model import Factor, Factors


class TestIdentifyKeyFactors(IsolatedAsyncioTestCase):
    async def test_get_key_factors_builds_messages_and_calls_client(self) -> None:
        expected = Factors(factors=[Factor(name="Cost", description="High cost")])
        client = Mock(responses_parse=Mock(return_value=expected))
        step = IdentifyKeyFactors(client, Path("checkpoint.bin"))

        result = await step.get_key_factors(" Research? ", ["doc one", "doc two"])

        self.assertEqual(result, expected)
        client.responses_parse.assert_called_once()
        _, kwargs = client.responses_parse.call_args
        self.assertIn("Research?", kwargs["instructions"])
        self.assertEqual(
            kwargs["messages"],
            [
                {
                    "role": "user",
                    "content": "<documents><document>doc one</document><document>doc two</document></documents>",
                },
                {
                    "role": "user",
                    "content": (
                        "Find the key factors to the research question from the text "
                        "provided"
                    ),
                },
            ],
        )
        self.assertIs(kwargs["response_format"], Factors)

    async def test_process_invokes_get_key_factors_for_document_list(self) -> None:
        expected = ["factors"]
        step = IdentifyKeyFactors(Mock(), Path("checkpoint.bin"))
        step.get_key_factors = AsyncMock(return_value=expected)

        result = await step.process("question", ["body-a", "body-b"])

        self.assertEqual(result, expected)
        step.get_key_factors.assert_awaited_once_with("question", ["body-a", "body-b"])
