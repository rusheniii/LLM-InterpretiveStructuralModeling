from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, call, patch

from llmism.model import (
    Chunk,
    Factor,
    SSIMFullGraphElement,
    SSIMFullGraphElements,
    SSIMKWiseElement,
    SSIMKWiseElements,
    StructuralSelfInteractionElement,
)
from llmism.StructuralSelfInteractMatrix import (
    SSIMBase,
    SSIMFullGraph,
    SSIMNoDataKWise as SSIMNoData,
    SSIMNoDataFullGraph,
    SSIMRagKwise,
)


def _factor(name: str) -> Factor:
    return Factor(name=name, description=f"{name} description")


def _edge(factor_i: str, factor_j: str, relationship: str = "V"):
    return StructuralSelfInteractionElement(
        factor_i=factor_i,
        factor_j=factor_j,
        relationship=relationship,
    )


class FakeSSIM(SSIMBase):
    async def _process(self, research_question, factors, **kwargs):
        return [_edge("A", "B")]


class InvalidSSIM(SSIMBase):
    async def _process(self, research_question, factors, **kwargs):
        return [_edge("A", "Missing")]


class TestStructuralSelfInteractMatrix(IsolatedAsyncioTestCase):
    def test_validate_edges_returns_valid_edges(self) -> None:
        expected = [_edge("A", "B")]
        step = SSIMBase(Mock(), Path("checkpoint.bin"), "prompt")

        result = step.validate_edges(expected, {"A", "B"})

        self.assertEqual(result, expected)

    def test_validate_edges_rejects_unknown_factor(self) -> None:
        step = SSIMBase(Mock(), Path("checkpoint.bin"), "prompt")

        with self.assertRaisesRegex(ValueError, "invalid factors"):
            step.validate_edges([_edge("A", "Missing")], {"A", "B"})

    async def test_process_validates_edges_from_process_hook(self) -> None:
        expected = {("A", "A"), ("A", "B"), ("B", "B")}
        step = FakeSSIM(Mock(), Path("checkpoint.bin"), "prompt")

        result = await step.process("question", [_factor("A"), _factor("B")])

        self.assertEqual(set(result.edges()), expected)

    async def test_process_reports_and_reraises_invalid_edges(self) -> None:
        step = InvalidSSIM(Mock(), Path("checkpoint.bin"), "prompt")

        with patch("llmism.StructuralSelfInteractMatrix.logger.exception") as log_exception:
            with self.assertRaisesRegex(ValueError, "invalid factors"):
                await step.process("question", [_factor("A"), _factor("B")])

        log_exception.assert_called_once()
        self.assertEqual(log_exception.call_args.args[0], "Failed to process SSIM: %s")
        self.assertIsInstance(log_exception.call_args.args[1], ValueError)
        self.assertEqual(log_exception.call_args.kwargs, {"exc_info": True})

    async def test_create_ssim_builds_no_data_prompt_and_calls_client(self) -> None:
        expected = SSIMKWiseElements(
            factors=[
                SSIMKWiseElement(
                    provided_factor_name="A",
                    factor_name="B",
                    relationship="V",
                )
            ]
        )
        client = Mock(inference=AsyncMock(return_value=expected))
        step = SSIMNoData(client, Path("checkpoint.bin"), "%s|%s|%s", batch=1)

        result = await step.create_ssim("question", _factor("A"), [_factor("B")])

        self.assertEqual(result, expected)
        client.inference.assert_awaited_once()
        _, kwargs = client.inference.call_args
        self.assertEqual(
            kwargs["instructions"],
            (
                "question|"
                "<provided_factor><provided_factor_name>A</provided_factor_name>"
                "</provided_factor>|"
                "<factors><factor><factor_name>B</factor_name></factor>"
                "</factors>"
            ),
        )
        self.assertEqual(
            kwargs["messages"],
            [
                {
                    "role": "user",
                    "content": (
                        "Follow the instructions and return a JSON object with the "
                        "specified fields."
                    ),
                },
            ],
        )
        self.assertIs(kwargs["response_format"], SSIMKWiseElements)

    async def test_get_ssim_no_data_batches_remaining_factors(self) -> None:
        expected = [("A", "B"), ("B", "A")]
        step = SSIMNoData(Mock(), Path("checkpoint.bin"), "%s|%s|%s", batch=1)
        step.create_ssim = AsyncMock(
            side_effect=[
                SSIMKWiseElements(
                    factors=[
                        SSIMKWiseElement(
                            provided_factor_name="A",
                            factor_name="B",
                            relationship="V",
                        )
                    ]
                ),
                SSIMKWiseElements(
                    factors=[
                        SSIMKWiseElement(
                            provided_factor_name="B",
                            factor_name="A",
                            relationship="V",
                        )
                    ]
                ),
            ]
        )
        factors = [_factor("A"), _factor("B")]

        result = await step.get_ssim_no_data("question", factors)

        self.assertEqual(
            [(edge.provided_factor_name, edge.factor_name) for edge in result.factors],
            expected,
        )
        self.assertEqual(
            step.create_ssim.await_args_list,
            [
                call("question", factors[0], (factors[1],)),
                call("question", factors[1], (factors[0],)),
            ],
        )

    async def test_no_data_process_converts_kwise_edges_to_structural_edges(
        self,
    ) -> None:
        expected = [("A", "B", "V")]
        step = SSIMNoData(Mock(), Path("checkpoint.bin"), "%s|%s|%s")
        step.get_ssim_no_data = AsyncMock(
            return_value=SSIMKWiseElements(
                factors=[
                    SSIMKWiseElement(
                        provided_factor_name="A",
                        factor_name="B",
                        relationship="V",
                    )
                ]
            )
        )

        result = await step._process("question", [_factor("A"), _factor("B")])

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result],
            expected,
        )
        step.get_ssim_no_data.assert_awaited_once()

    async def test_rag_get_embedding_delegates_to_embedding_client(self) -> None:
        expected = [0.1]
        client = Mock(get_embedding=AsyncMock(return_value=expected))
        step = SSIMRagKwise(
            client,
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )

        result = await step.get_embedding("text")

        self.assertEqual(result, expected)
        client.get_embedding.assert_awaited_once_with("text")

    def test_rag_build_supporting_documents_wraps_chunks(self) -> None:
        expected = "<documents><document>a</document><document>b</document></documents>"
        step = SSIMRagKwise(
            Mock(),
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )

        result = step._build_supporting_documents(["a", "b"])

        self.assertEqual(result, expected)

    async def test_rag_create_ssim_documents_calls_client_with_documents(
        self,
    ) -> None:
        expected = SSIMKWiseElements(
            factors=[
                SSIMKWiseElement(
                    provided_factor_name="A",
                    factor_name="B",
                    relationship="V",
                )
            ]
        )
        client = Mock(inference=AsyncMock(return_value=expected))
        step = SSIMRagKwise(
            client,
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )

        result = await step.create_ssim_with_documents(
            "question",
            _factor("A"),
            [_factor("B")],
            ["support"],
        )

        self.assertEqual(result, expected)
        client.inference.assert_awaited_once()
        _, kwargs = client.inference.call_args
        self.assertEqual(kwargs["messages"][0]["content"], "<documents><document>support</document></documents>")
        self.assertIn("<provided_factor_name>A</provided_factor_name>", kwargs["instructions"])
        self.assertIs(kwargs["response_format"], SSIMKWiseElements)

    async def test_rag_get_causal_edges_with_nodata_keeps_v_relationships(
        self,
    ) -> None:
        expected = {"B": {"A"}, "A": {"C"}}
        step = SSIMRagKwise(
            Mock(),
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )
        step.get_ssim_no_data = AsyncMock(
            return_value=SimpleNamespace(
                factors=[
                    SimpleNamespace(provided_factor_name="B", factor_name="A", relationship="V"),
                    SimpleNamespace(provided_factor_name="A", factor_name="C", relationship="V"),
                    SimpleNamespace(provided_factor_name="A", factor_name="D", relationship="O"),
                ]
            )
        )

        result = await step.get_causal_edges_with_nodata(
            "question",
            [_factor("A"), _factor("B"), _factor("C")],
        )

        self.assertEqual(dict(result), expected)
        step.get_ssim_no_data.assert_awaited_once()

    async def test_rag_get_top_chunks_queries_vector_store_for_each_factor(
        self,
    ) -> None:
        expected = ["chunk-0", "chunk-1"]
        partitioned = [str(index) for index in range(7)]
        chunks = [Chunk(text=f"chunk-{index}", metadata={}, embedding=[1.0]) for index in range(2)]
        vector_store = Mock(get_top_chunks=Mock(return_value=chunks))
        client = Mock(get_embedding=AsyncMock(return_value=[0.1]))
        step = SSIMRagKwise(
            client,
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )

        result = await step.get_top_chunks(
            "question",
            _factor("Base"),
            partitioned,
            vector_store,
        )

        self.assertEqual(result[:2], expected)
        self.assertLessEqual(len(result), 14)
        client.get_embedding.assert_any_await("question\nBase\n0")
        vector_store.get_top_chunks.assert_any_call(query=[0.1], top_k=3)

    async def test_rag_get_causal_edges_with_data_combines_batches(self) -> None:
        expected = [("A", "B")]
        step = SSIMRagKwise(
            Mock(),
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
            batch=1,
        )
        step.get_top_chunks = AsyncMock(return_value=["support"])
        factor = _factor("A")
        vector_store = Mock()
        step.create_ssim_with_documents = AsyncMock(
            return_value=SSIMKWiseElements(
                factors=[
                    SSIMKWiseElement(
                        provided_factor_name="A",
                        factor_name="B",
                        relationship="V",
                    )
                ]
            )
        )
        undirected = defaultdict(set, {"A": {"B"}})

        result = await step.get_causal_edges_with_data(
            "question",
            [factor],
            vector_store,
            undirected,
        )

        self.assertEqual(
            [(edge.provided_factor_name, edge.factor_name) for edge in result.factors],
            expected,
        )
        step.get_top_chunks.assert_awaited_once_with(
            "question",
            factor,
            ("B",),
            vector_store,
        )
        step.create_ssim_with_documents.assert_awaited_once_with(
            "question",
            factor,
            ("B",),
            ["support"],
        )

    async def test_rag_process_converts_directed_edges_to_structural_edges(
        self,
    ) -> None:
        expected = [("A", "B", "V")]
        step = SSIMRagKwise(
            Mock(),
            Path("checkpoint.bin"),
            "%s|%s|%s",
            "%s|%s|%s",
        )
        step.get_causal_edges_with_nodata = AsyncMock(return_value={"A": {"B"}})
        step.get_causal_edges_with_data = AsyncMock(
            return_value=SSIMKWiseElements(
                factors=[
                    SSIMKWiseElement(
                        provided_factor_name="A",
                        factor_name="B",
                        relationship="V",
                    )
                ]
            )
        )

        result = await step._process("question", [_factor("A"), _factor("B")], Mock())

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result],
            expected,
        )
        step.get_causal_edges_with_nodata.assert_awaited_once()
        step.get_causal_edges_with_data.assert_awaited_once()

    async def test_full_graph_no_data_create_ssim_calls_client(self) -> None:
        expected = SSIMFullGraphElements(
            factors=[SSIMFullGraphElement(factor_i="A", factor_j="B", relationship="V")]
        )
        client = Mock(inference=AsyncMock(return_value=expected))
        step = SSIMNoDataFullGraph(client, Path("checkpoint.bin"), "%s|%s")

        result = await step.create_ssim("question", [_factor("A"), _factor("B")])

        self.assertEqual(result, expected)
        client.inference.assert_awaited_once()
        _, kwargs = client.inference.call_args
        self.assertIn("<factors>", kwargs["instructions"])
        self.assertIs(kwargs["response_format"], SSIMFullGraphElements)

    async def test_full_graph_no_data_process_converts_full_graph_edges(
        self,
    ) -> None:
        expected = [("A", "B", "V")]
        step = SSIMNoDataFullGraph(Mock(), Path("checkpoint.bin"), "%s|%s")
        step.create_ssim = AsyncMock(
            return_value=SSIMFullGraphElements(
                factors=[
                    SSIMFullGraphElement(factor_i="A", factor_j="B", relationship="V")
                ]
            )
        )

        result = await step._process("question", [_factor("A"), _factor("B")])

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result],
            expected,
        )
        step.create_ssim.assert_awaited_once()

    async def test_full_graph_get_embedding_delegates_to_embedding_client(
        self,
    ) -> None:
        expected = [0.1]
        client = Mock(get_embedding=AsyncMock(return_value=expected))
        step = SSIMFullGraph(
            client,
            Path("checkpoint.bin"),
            "%s|%s",
            "%s|%s|%s",
        )

        result = await step.get_embedding("text")

        self.assertEqual(result, expected)
        client.get_embedding.assert_awaited_once_with("text")

    def test_full_graph_build_supporting_documents_wraps_chunks(self) -> None:
        expected = "<documents><document>a</document></documents>"
        step = SSIMFullGraph(Mock(), Path("checkpoint.bin"), "%s|%s", "%s|%s|%s")

        result = step._build_supporting_documents(["a"])

        self.assertEqual(result, expected)

    async def test_full_graph_create_ssim_documents_calls_client(self) -> None:
        expected = SSIMKWiseElements(
            factors=[
                SSIMKWiseElement(
                    provided_factor_name="A",
                    factor_name="B",
                    relationship="V",
                )
            ]
        )
        client = Mock(inference=AsyncMock(return_value=expected))
        step = SSIMFullGraph(client, Path("checkpoint.bin"), "%s|%s", "%s|%s|%s")

        result = await step.create_ssim_with_documents(
            "question",
            _factor("A"),
            [_factor("B")],
            ["support"],
        )

        self.assertEqual(result, expected)
        client.inference.assert_awaited_once()
        _, kwargs = client.inference.call_args
        self.assertEqual(kwargs["messages"][0]["content"], "<documents><document>support</document></documents>")
        self.assertIs(kwargs["response_format"], SSIMKWiseElements)

    async def test_full_graph_get_causal_edges_with_nodata_uses_full_graph_edges(
        self,
    ) -> None:
        expected = {"B": {"A"}, "A": {"C"}}
        step = SSIMFullGraph(Mock(), Path("checkpoint.bin"), "%s|%s", "%s|%s|%s")
        step.create_ssim = AsyncMock(
            return_value=SSIMFullGraphElements(
                factors=[
                    SSIMFullGraphElement(factor_i="B", factor_j="A", relationship="V"),
                    SSIMFullGraphElement(factor_i="A", factor_j="C", relationship="V"),
                    SSIMFullGraphElement(factor_i="A", factor_j="D", relationship="O"),
                ]
            )
        )

        result = await step.get_causal_edges_with_nodata(
            "question",
            [_factor("A"), _factor("B")],
        )

        self.assertEqual(dict(result), expected)
        step.create_ssim.assert_awaited_once()

    async def test_full_graph_get_causal_edges_with_data_queries_chunks(
        self,
    ) -> None:
        expected = [("A", "B")]
        vector_store = Mock()
        step = SSIMFullGraph(Mock(), Path("checkpoint.bin"), "%s|%s", "%s|%s|%s")
        step.get_top_chunks = AsyncMock(return_value=["support"])
        step.create_ssim_with_documents = AsyncMock(
            return_value=SSIMKWiseElements(
                factors=[
                    SSIMKWiseElement(
                        provided_factor_name="A",
                        factor_name="B",
                        relationship="V",
                    )
                ]
            )
        )
        undirected = defaultdict(set, {"A": {"B"}})

        result = await step.get_causal_edges_with_data(
            "question",
            [_factor("A")],
            vector_store,
            undirected,
        )

        self.assertEqual(
            [(edge.provided_factor_name, edge.factor_name) for edge in result.factors],
            expected,
        )
        step.get_top_chunks.assert_awaited_once_with(
            "question",
            _factor("A"),
            {"B"},
            vector_store,
        )
        step.create_ssim_with_documents.assert_awaited_once_with(
            "question",
            _factor("A"),
            {"B"},
            ["support"],
        )

    async def test_full_graph_process_converts_directed_edges(self) -> None:
        expected = [("A", "B", "V")]
        step = SSIMFullGraph(Mock(), Path("checkpoint.bin"), "%s|%s", "%s|%s|%s")
        step.get_causal_edges_with_nodata = AsyncMock(return_value={"A": {"B"}})
        step.get_causal_edges_with_data = AsyncMock(
            return_value=SSIMFullGraphElements(
                factors=[
                    SSIMFullGraphElement(factor_i="A", factor_j="B", relationship="V")
                ]
            )
        )

        result = await step._process("question", [_factor("A"), _factor("B")], Mock())

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result],
            expected,
        )
        step.get_causal_edges_with_nodata.assert_awaited_once()
        step.get_causal_edges_with_data.assert_awaited_once()
