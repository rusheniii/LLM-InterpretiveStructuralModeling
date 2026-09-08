from unittest import TestCase

from llmism.model import (
    Chunk,
    Factor,
    SSIMFullGraphElement,
    SSIMFullGraphElements,
    SSIMKWiseElement,
    SSIMKWiseElements,
    SimpleVectorStore,
    StructuralSelfInteractionElement,
    StructuralSelfInteractionElements,
    cosine,
)
from llmism.ssim_utils import (
    document_chunk_to_xml,
    document_chunks_to_xml,
    factor_to_xml,
    factors_to_xml,
    provided_factor_to_xml,
)


class TestModelAndUtils(TestCase):
    def test_cosine_returns_normalized_dot_product(self) -> None:
        expected = 0.0

        result = cosine([1.0, 0.0], [0.0, 1.0])

        self.assertEqual(result, expected)

    def test_simple_vector_store_add_chunk_extends_existing_chunks(self) -> None:
        expected = ["first", "second"]
        store = SimpleVectorStore(
            name="store",
            chunks=[Chunk(text="first", metadata={}, embedding=[1.0, 0.0])],
        )
        new_chunk = Chunk(text="second", metadata={}, embedding=[0.0, 1.0])

        store.add_chunk(new_chunk)

        self.assertEqual([chunk.text for chunk in store.chunks], expected)

    def test_simple_vector_store_get_top_chunks_orders_by_cosine(self) -> None:
        expected = ["near", "middle"]
        store = SimpleVectorStore(
            name="store",
            chunks=[
                Chunk(text="far", metadata={}, embedding=[0.0, 1.0]),
                Chunk(text="near", metadata={}, embedding=[1.0, 0.0]),
                Chunk(text="middle", metadata={}, embedding=[0.5, 0.5]),
            ],
        )

        result = store.get_top_chunks(query=[1.0, 0.0], top_k=2)

        self.assertEqual([chunk.text for chunk in result], expected)

    def test_simple_vector_store_get_top_chunks_caps_to_available_chunks(self) -> None:
        expected = ["only"]
        store = SimpleVectorStore(
            name="store",
            chunks=[Chunk(text="only", metadata={}, embedding=[1.0, 0.0])],
        )

        result = store.get_top_chunks(query=[1.0, 0.0], top_k=5)

        self.assertEqual([chunk.text for chunk in result], expected)

    def test_factor_defaults_description_to_none(self) -> None:
        expected = {"name": "A", "description": None}

        result = Factor(name="A")

        self.assertEqual(result.model_dump(), expected)

    def test_structural_self_interaction_element_accepts_full_graph_aliases(
        self,
    ) -> None:
        expected = {
            "factor_i": "A",
            "factor_j": "B",
            "relationship": "V",
        }

        result = StructuralSelfInteractionElement(
            factor_i="A",
            factor_j="B",
            relationship="V",
        )

        self.assertEqual(result.model_dump(), expected)

    def test_structural_self_interaction_element_accepts_kwise_aliases(self) -> None:
        expected = {
            "factor_i": "A",
            "factor_j": "B",
            "relationship": "V",
        }

        result = StructuralSelfInteractionElement(
            provided_factor_name="A",
            factor_name="B",
            relationship="V",
        )

        self.assertEqual(result.model_dump(), expected)

    def test_structural_self_interaction_elements_validate_kwise_dump(self) -> None:
        expected = [("A", "B", "V")]
        kwise = SSIMKWiseElements(
            factors=[
                SSIMKWiseElement(
                    provided_factor_name="A",
                    factor_name="B",
                    relationship="V",
                )
            ]
        )

        result = StructuralSelfInteractionElements.model_validate(kwise.model_dump())

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result.factors],
            expected,
        )

    def test_structural_self_interaction_elements_validate_full_graph_dump(
        self,
    ) -> None:
        expected = [("A", "B", "V")]
        full_graph = SSIMFullGraphElements(
            factors=[
                SSIMFullGraphElement(
                    factor_i="A",
                    factor_j="B",
                    relationship="V",
                )
            ]
        )

        result = StructuralSelfInteractionElements.model_validate(
            full_graph.model_dump()
        )

        self.assertEqual(
            [(edge.factor_i, edge.factor_j, edge.relationship) for edge in result.factors],
            expected,
        )

    def test_document_chunk_to_xml_wraps_text(self) -> None:
        expected = "<document>chunk</document>"

        result = document_chunk_to_xml("chunk")

        self.assertEqual(result, expected)

    def test_document_chunks_to_xml_wraps_all_chunks(self) -> None:
        expected = "<documents><document>a</document><document>b</document></documents>"

        result = document_chunks_to_xml(["a", "b"])

        self.assertEqual(result, expected)

    def test_factor_to_xml_wraps_factor_name(self) -> None:
        expected = (
            "<factor><factor_name>name='Cost' description='High cost'"
            "</factor_name></factor>"
        )

        result = factor_to_xml(Factor(name="Cost", description="High cost"))

        self.assertEqual(result, expected)

    def test_factors_to_xml_wraps_all_factors(self) -> None:
        expected = (
            "<factors>"
            "<factor><factor_name>A</factor_name></factor>"
            "<factor><factor_name>B</factor_name></factor>"
            "</factors>"
        )

        result = factors_to_xml([Factor(name="A"), Factor(name="B")])

        self.assertEqual(result, expected)

    def test_provided_factor_to_xml_wraps_factor_name(self) -> None:
        expected = (
            "<provided_factor>"
            "<provided_factor_name>Cost</provided_factor_name>"
            "</provided_factor>"
        )

        result = provided_factor_to_xml(Factor(name="Cost"))

        self.assertEqual(result, expected)
