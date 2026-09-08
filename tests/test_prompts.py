from unittest import TestCase

from llmism.prompts.IdentifyKeyFactorsPrompt import KEY_FACTORS_PROMPT
from llmism.prompts.SSIMEdgeDirectionPrompt import EDGE_DIRECTION_PROMPT
from llmism.prompts.SSIMNoData import SSIM_NO_DATA_PROMPT
from llmism.prompts.SSIMNoDataFullGraph import SSIM_FULL_GRAPH


class TestPrompts(TestCase):
    def test_key_factors_prompt_contains_research_question_placeholder(self) -> None:
        expected = "<research_question>question</research_question>"

        result = KEY_FACTORS_PROMPT % "question"

        self.assertIn(expected, result)

    def test_edge_direction_prompt_contains_research_question_and_factors(
        self,
    ) -> None:
        expected = ["<research_question>question</research_question>", "<provided>A</provided>", "<factors></factors>"]

        result = EDGE_DIRECTION_PROMPT % (
            "question",
            "<provided>A</provided>",
            "<factors></factors>",
        )

        for value in expected:
            self.assertIn(value, result)

    def test_ssim_no_data_prompt_contains_factor_context(self) -> None:
        expected = ["<research_question>question</research_question>", "<provided>A</provided>", "<factors></factors>"]

        result = SSIM_NO_DATA_PROMPT % (
            "question",
            "<provided>A</provided>",
            "<factors></factors>",
        )

        for value in expected:
            self.assertIn(value, result)

    def test_ssim_full_graph_prompt_contains_factor_context(self) -> None:
        expected = ["<research_question>question</research_question>", "<factors></factors>"]

        result = SSIM_FULL_GRAPH % ("question", "<factors></factors>")

        for value in expected:
            self.assertIn(value, result)
