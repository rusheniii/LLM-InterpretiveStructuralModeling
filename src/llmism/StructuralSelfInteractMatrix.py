import itertools
import logging
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import networkx as nx
from .ISMBaseStep import ISMBaseStep
from .LLMClient import LLMClient, append_message
from .model import (
    EmbeddedFactor,
    Factor,
    SSIMFullGraphElements,
    SSIMKWiseElements,
    SimpleVectorStore,
    StructuralSelfInteractionElement,
    StructuralSelfInteractionElements,
)
from .ssim_utils import (
    document_chunks_to_xml,
    factor_names_to_xml,
    factors_to_xml,
    provided_factor_to_xml,
)


logger = logging.getLogger(__name__)


class SSIMBase(ISMBaseStep):
    def __init__(self, client: LLMClient, checkpoint: Path, prompt: str):
        super().__init__(client, checkpoint)
        self.prompt = prompt

    def build_transitive_closure(self, graph: nx.DiGraph):
        fw = nx.floyd_warshall(graph)
        transitive_closure = nx.DiGraph()
        for i in fw:
            for j in fw[i]:
                if fw[i][j] != float('inf'):
                    transitive_closure.add_edge(i, j)
        for node in transitive_closure.nodes:
            if not transitive_closure.has_edge(node, node):
                transitive_closure.add_edge(node, node)
        return transitive_closure

    def validate_edges(
        self,
        edges: list[StructuralSelfInteractionElement],
        validated_factor_names: set[str],
    ) -> list[StructuralSelfInteractionElement]:
        for edge in edges:
            if (
                edge.factor_i not in validated_factor_names
                or edge.factor_j not in validated_factor_names
            ):
                raise ValueError(
                    f"Edge contains invalid factors: {edge.factor_i}, "
                    f"{edge.factor_j}"
                )
        return edges

    def transform_to_nx_graph(self, edges: list[StructuralSelfInteractionElement], factors: list[Factor]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for edge in edges:
            if edge.relationship == 'V':
                graph.add_edge(edge.factor_i, edge.factor_j)
            #elif edge.relationship == 'A':
            #    graph.add_edge(edge.factor_j, edge.factor_i)
        for factor in factors:
            if not graph.has_edge(factor.name, factor.name):
                graph.add_edge(factor.name, factor.name)
        return graph

    async def process(
        self,
        research_question,
        factors,
        **kwargs,
    ) -> nx.DiGraph:
        validated_factor_names = {f.name for f in factors}
        try:
            edges = await self._process(research_question, factors, **kwargs)
            self.validate_edges(edges, validated_factor_names)
        except Exception as e:
            logger.exception("Failed to process SSIM: %s", e, exc_info=True)
            raise e
        return self.build_transitive_closure(
            self.transform_to_nx_graph(edges, factors)
        )


class RagProtocol(Protocol):
    client: LLMClient
    edge_direction_prompt: str

class SSIMRagMixin(object):

    async def get_embedding(self: RagProtocol, text: str):
        return await self.client.get_embedding(text)

    async def get_top_text_for_factors(
        self,
        research_question: str,
        factor_i: Factor,
        factor_j: str,
        vector_store: SimpleVectorStore,
    ):
        embedding = await self.get_embedding(
            f"{research_question}\n{factor_i.name}\n{factor_j}"
        )
        top_k = 3
        texts = []
        for chunk in vector_store.get_top_chunks(query=embedding, top_k=top_k):
            texts.append(chunk.text)
        return texts

    async def get_top_chunks(
        self,
        research_question: str,
        factor: Factor,
        factors: list[str],
        vector_store: SimpleVectorStore,
        top_k: int = 14,
    ):
        top_chunks = []
        for rfactor in factors:
            top_chunks.extend(
                await self.get_top_text_for_factors(
                    research_question,
                    factor,
                    rfactor,
                    vector_store,
                )
            )
        return top_chunks[:min(len(top_chunks), top_k)]

    def _build_supporting_documents(
            self: RagProtocol,
            text_chunks: list[str]
    ) -> str:
        return document_chunks_to_xml(text_chunks)

    async def create_ssim_with_documents(
        self: RagProtocol,
        research_question: str,
        factor_i: Factor,
        factors: list[str],
        text_chunks: list[str],
    ):
        logger.info(
            "Creating SSIM with %s documents for research question: %s",
            len(text_chunks),
            research_question
        )
        messages = []
        instructions = self.edge_direction_prompt % (
            research_question,
            provided_factor_to_xml(factor_i),
            factor_names_to_xml(factors),
        )
        full_text = self._build_supporting_documents(text_chunks)
        user_prompt = (
            "Follow the instructions and return a JSON object with the "
            "specified fields."
        )
        append_message("user", full_text, messages)
        append_message("user", user_prompt, messages)
        response = await self.client.inference(
            instructions=instructions,
            messages=messages,
            temperature=0.0,
            response_format=SSIMKWiseElements,
        )
        return response


class SSIMNoDataKWise(SSIMBase):
    def __init__(
        self,
        client: LLMClient,
        checkpoint: Path,
        prompt: str,
        batch: int = 15,
    ):
        super().__init__(client, checkpoint, prompt)
        self.batch = batch

    async def create_ssim(
        self,
        research_question: str,
        factor_i: Factor,
        factors: list[Factor],
    ):
        messages = []
        instructions = self.prompt % (
            research_question,
            provided_factor_to_xml(factor_i),
            factors_to_xml(factors),
        )
        user_prompt = (
            "Follow the instructions and return a JSON object with the "
            "specified fields."
        )
        append_message("user", user_prompt, messages)
        response = await self.client.inference(
            instructions=instructions,
            messages=messages,
            response_format=SSIMKWiseElements,
        )
        return response

    async def get_ssim_no_data(
        self,
        research_question: str,
        factors: list[Factor],
    ):
        ssim_edges = []
        for factor in factors:
            remaining_factors = factors[:]
            remaining_factors.remove(factor)
            for partitioned_factors in itertools.batched(
                remaining_factors,
                self.batch,
            ):
                edges = await self.create_ssim(
                    research_question,
                    factor,
                    partitioned_factors,
                )
                logger.warning(
                    "Discovered %s edges for %s",
                    len(edges.factors),
                    factor.name,
                )
                ssim_edges.extend(edges.factors)
        return SSIMKWiseElements(factors=ssim_edges)

    async def _process(self, research_question: str, factors: list[Factor]):
        full_graph = await self.get_ssim_no_data(research_question, factors)
        return (
            StructuralSelfInteractionElements
            .model_validate(full_graph.model_dump())
            .factors
        )


class SSIMRagKwise(SSIMNoDataKWise, SSIMRagMixin):
    def __init__(
        self,
        client: LLMClient,
        checkpoint: Path,
        prompt: str,
        edge_direction_prompt: str,
        batch: int = 15,
    ):
        super().__init__(client, checkpoint, prompt, batch)
        self.edge_direction_prompt = edge_direction_prompt

    async def get_causal_edges_with_nodata(
        self,
        research_question: str,
        factors: list[Factor],
    ):
        ssim_edges = defaultdict(set)
        edges = await self.get_ssim_no_data(research_question, factors)
        for edge in edges.factors:
            factor_i = edge.provided_factor_name
            factor_j = edge.factor_name
            if edge.relationship == "V":
                ssim_edges[factor_i].add(factor_j)
        return ssim_edges

    async def get_causal_edges_with_data(
        self,
        research_question: str,
        factors: list[Factor],
        vector_store: SimpleVectorStore,
        undirected_edges: dict[str, set[str]],
    ):
        ssim_edges = []
        for factor in factors:
            for partitioned_factors in itertools.batched(
                undirected_edges[factor.name],
                self.batch,
            ):
                top_chunks = await self.get_top_chunks(
                    research_question,
                    factor,
                    partitioned_factors,
                    vector_store,
                )
                edges = await self.create_ssim_with_documents(
                    research_question,
                    factor,
                    partitioned_factors,
                    top_chunks,
                )
                logger.info(
                    "Discovered %s edges for %s",
                    len(edges.factors),
                    factor.name,
                )
                ssim_edges.extend(edges.factors)
        return SSIMKWiseElements(factors=ssim_edges)

    async def _process(
        self,
        research_question: str,
        factors: list[EmbeddedFactor],
        vector_store: SimpleVectorStore,
    ):
        ssim_edges = await self.get_causal_edges_with_nodata(
            research_question,
            factors,
        )
        full_graph = await self.get_causal_edges_with_data(
            research_question,
            factors,
            vector_store,
            ssim_edges,
        )
        return (
            StructuralSelfInteractionElements
            .model_validate(full_graph.model_dump())
            .factors
        )


class SSIMNoDataFullGraph(SSIMBase):

    async def create_ssim(self, research_question: str, factors: list[Factor]):
        messages = []
        instructions = self.prompt % (
            research_question,
            factors_to_xml(factors),
        )
        user_prompt = (
            "Follow the instructions and return a JSON object with the "
            "specified fields."
        )
        append_message("user", user_prompt, messages)
        response = await self.client.inference(
            instructions=instructions,
            messages=messages,
            temperature=0.0,
            response_format=SSIMFullGraphElements,
        )
        return response

    async def _process(self, research_question: str, factors: list[Factor]):
        full_graph = await self.create_ssim(research_question, factors)
        return (
            StructuralSelfInteractionElements
            .model_validate(full_graph.model_dump())
            .factors
        )


class SSIMFullGraph(SSIMNoDataFullGraph, SSIMRagMixin):
    def __init__(
        self,
        client: LLMClient,
        checkpoint: Path,
        prompt: str,
        edge_direction_prompt: str,
    ):
        super().__init__(client, checkpoint, prompt)
        self.edge_direction_prompt = edge_direction_prompt

    async def get_causal_edges_with_nodata(
        self,
        research_question: str,
        factors: list[Factor],
    ):
        ssim_edges = defaultdict(set)
        edges = await self.create_ssim(research_question, factors)
        for edge in edges.factors:
            factor_i = edge.factor_i
            factor_j = edge.factor_j
            if edge.relationship == "V":
                ssim_edges[factor_i].add(factor_j)
        return ssim_edges

    async def get_causal_edges_with_data(
        self,
        research_question: str,
        factors: list[Factor],
        vector_store: SimpleVectorStore,
        nodata_edges: dict[str, set[str]],
    ):
        ssim_edges = []
        for factor in factors:
            causal_factors = nodata_edges[factor.name]
            top_chunks = await self.get_top_chunks(
                research_question,
                factor,
                causal_factors,
                vector_store,
            )
            edges = await self.create_ssim_with_documents(
                research_question,
                factor,
                causal_factors,
                top_chunks,
            )
            ssim_edges.extend(edges.factors)
        return SSIMKWiseElements(factors=ssim_edges)

    async def _process(
        self,
        research_question: str,
        factors: list[Factor],
        vector_store: SimpleVectorStore,
    ):
        logger.info("Processing SSIM Full Graph")
        nodata_edges = await self.get_causal_edges_with_nodata(
            research_question,
            factors,
        )
        full_graph = await self.get_causal_edges_with_data(
            research_question,
            factors,
            vector_store,
            nodata_edges,
        )
        return (
            StructuralSelfInteractionElements
            .model_validate(full_graph.model_dump())
            .factors
        )
