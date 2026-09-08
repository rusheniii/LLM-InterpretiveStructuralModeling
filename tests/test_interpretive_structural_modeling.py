from pathlib import Path
import importlib
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from llmism.InterpretiveStructuralModeling import InterpretiveStructuralModeling
from llmism.model import Approach

ism_module = importlib.import_module("llmism.InterpretiveStructuralModeling")


class TestInterpretiveStructuralModeling(TestCase):
    def test_initialization_wraps_openai_client(self) -> None:
        openai_client = Mock()
        wrapped_client = Mock()

        with patch.object(ism_module, "LLMClient", return_value=wrapped_client) as client_class:
            result = InterpretiveStructuralModeling(
                openai_client,
                "chat-model",
                "embedding-model",
                Approach.ROWWISE,
                batch=4,
            )

        self.assertIs(result.client, wrapped_client)
        self.assertEqual(result.method, Approach.ROWWISE)
        self.assertEqual(result.batch, 4)
        client_class.assert_called_once_with(
            openai_client,
            "chat-model",
            "embedding-model",
            responses_api_supported=True,
        )

    def test_get_approach_selects_no_data_full_graph(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")
        implementation = Mock()

        with patch.object(ism_module, "SSIMNoDataFullGraph", return_value=implementation) as approach_class:
            result = model._get_approach([], ["Cost"], Path("checkpoint.bin"))

        self.assertIs(result, implementation)
        approach_class.assert_called_once_with(
            model.client,
            Path("checkpoint.bin"),
            prompt=ism_module.SSIM_FULL_GRAPH,
        )

    def test_get_approach_selects_full_graph_with_documents(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")
        implementation = Mock()

        with patch.object(ism_module, "SSIMFullGraph", return_value=implementation) as approach_class:
            result = model._get_approach([Mock()], ["Cost"], Path("checkpoint.bin"))

        self.assertIs(result, implementation)
        approach_class.assert_called_once_with(
            model.client,
            Path("checkpoint.bin"),
            prompt=ism_module.SSIM_FULL_GRAPH,
            edge_direction_prompt=ism_module.EDGE_DIRECTION_PROMPT,
        )

    def test_get_approach_selects_no_data_rowwise(self) -> None:
        model = InterpretiveStructuralModeling(
            Mock(), "chat-model", "embedding-model", Approach.ROWWISE, batch=4
        )
        implementation = Mock()

        with patch.object(ism_module, "SSIMNoDataKWise", return_value=implementation) as approach_class:
            result = model._get_approach([], ["Cost"], Path("checkpoint.bin"))

        self.assertIs(result, implementation)
        approach_class.assert_called_once_with(
            model.client,
            Path("checkpoint.bin"),
            prompt=ism_module.SSIM_NO_DATA_PROMPT,
            batch=1,
        )

    def test_get_approach_selects_rag_rowwise_with_documents(self) -> None:
        model = InterpretiveStructuralModeling(
            Mock(), "chat-model", "embedding-model", Approach.ROWWISE, batch=4
        )
        implementation = Mock()

        with patch.object(ism_module, "SSIMRagKwise", return_value=implementation) as approach_class:
            result = model._get_approach([Mock()], ["Cost"], Path("checkpoint.bin"))

        self.assertIs(result, implementation)
        approach_class.assert_called_once_with(
            model.client,
            Path("checkpoint.bin"),
            prompt=ism_module.SSIM_NO_DATA_PROMPT,
            edge_direction_prompt=ism_module.EDGE_DIRECTION_PROMPT,
            batch=1,
        )

    def test_get_approach_rejects_unsupported_method(self) -> None:
        model = InterpretiveStructuralModeling(
            Mock(), "chat-model", "embedding-model", "unsupported"
        )

        with self.assertRaisesRegex(ValueError, "Unsupported method: unsupported"):
            model._get_approach([Mock()], ["Cost"])

    def test_convert_key_factors_creates_factors_without_descriptions(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")

        result = model._convert_key_factors(["Cost", "Risk"])

        self.assertEqual([factor.name for factor in result], ["Cost", "Risk"])
        self.assertEqual([factor.description for factor in result], [None, None])


class TestInterpretiveStructuralModelingAsync(IsolatedAsyncioTestCase):
    async def test_embed_supporting_documents_keeps_successful_embeddings(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")
        model.client.get_embedding = AsyncMock(side_effect=[[0.1], RuntimeError("unavailable")])

        with patch("builtins.print") as print_mock:
            result = await model._embed_supporting_documents(["first", "second"])

        self.assertEqual(result.name, "store")
        self.assertEqual([(chunk.text, chunk.embedding) for chunk in result.chunks], [("first", [0.1])])
        model.client.get_embedding.assert_any_await("first")
        model.client.get_embedding.assert_any_await("second")
        print_mock.assert_called_once_with("Error embedding text: unavailable")

    async def test_get_structural_model_omits_store_when_no_embeddings_exist(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")
        model._embed_supporting_documents = AsyncMock(return_value=Mock(chunks=[]))
        approach = Mock(process_and_save=AsyncMock(return_value="graph"))
        model._get_approach = Mock(return_value=approach)

        result = await model.get_structural_model("question", ["Cost"], ["document"], "checkpoint.bin")

        self.assertEqual(result, "graph")
        model._embed_supporting_documents.assert_awaited_once_with(["document"])
        model._get_approach.assert_called_once_with([], ["Cost"], "checkpoint.bin")
        approach.process_and_save.assert_awaited_once_with(
            research_question="question",
            factors=model._convert_key_factors(["Cost"]),
        )

    async def test_get_structural_model_passes_store_when_embeddings_exist(self) -> None:
        model = InterpretiveStructuralModeling(Mock(), "chat-model", "embedding-model")
        vector_store = Mock(chunks=[Mock()])
        model._embed_supporting_documents = AsyncMock(return_value=vector_store)
        approach = Mock(process_and_save=AsyncMock(return_value="graph"))
        model._get_approach = Mock(return_value=approach)

        result = await model.get_structural_model("question", ["Cost"], ["document"], "checkpoint.bin")

        self.assertEqual(result, "graph")
        model._embed_supporting_documents.assert_awaited_once_with(["document"])
        model._get_approach.assert_called_once_with(vector_store.chunks, ["Cost"], "checkpoint.bin")
        approach.process_and_save.assert_awaited_once_with(
            research_question="question",
            factors=model._convert_key_factors(["Cost"]),
            vector_store=vector_store,
        )