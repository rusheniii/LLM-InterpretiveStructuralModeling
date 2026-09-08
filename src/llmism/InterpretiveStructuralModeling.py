from openai import AsyncOpenAI
from .model import Approach, Chunk, Factor, SimpleVectorStore

from .LLMClient import LLMClient
from .prompts.SSIMEdgeDirectionPrompt import EDGE_DIRECTION_PROMPT
from .prompts.SSIMNoData import SSIM_NO_DATA_PROMPT
from .prompts.SSIMNoDataFullGraph import SSIM_FULL_GRAPH
from .StructuralSelfInteractMatrix import (
    SSIMFullGraph,
    SSIMNoDataFullGraph,
    SSIMNoDataKWise,
    SSIMRagKwise,
)
import logging

logger = logging.getLogger(__name__)

class InterpretiveStructuralModeling:
    def __init__(self, client: AsyncOpenAI, model: str, embedding_model: str, method: Approach = Approach.FULL, batch = 15, responses_api_supported: bool = True):
        self.client = LLMClient(client, model, embedding_model, responses_api_supported=responses_api_supported)
        self.method = method
        self.batch = batch

    def _get_approach(self, supporting_documents: list[str], factors: list[str], checkpoint: str = None):
        if self.method == Approach.ROWWISE:
            self.batch = len(factors)
        if self.method == Approach.FULL and len(supporting_documents) == 0:
            logger.info("Using SSIMNoDataFullGraph approach as no supporting documents are provided.")
            return SSIMNoDataFullGraph(
                self.client,
                checkpoint,
                prompt=SSIM_FULL_GRAPH,
            )
        elif self.method == Approach.FULL and len(supporting_documents) > 0:
            logger.info("Using SSIMFullGraph approach as supporting documents are provided.")
            return SSIMFullGraph(
                self.client,
                checkpoint,
                prompt=SSIM_FULL_GRAPH,
                edge_direction_prompt=EDGE_DIRECTION_PROMPT,
            )
        elif (self.method == Approach.ROWWISE or self.method == Approach.KWISE) and len(supporting_documents) == 0:
            logger.info("Using SSIMNoDataKWise approach as no supporting documents are provided.")
            return SSIMNoDataKWise(
                self.client,
                checkpoint,
                prompt=SSIM_NO_DATA_PROMPT,
                batch=self.batch,
            )
        elif (self.method == Approach.ROWWISE or self.method == Approach.KWISE) and len(supporting_documents) > 0:
            logger.info("Using SSIMRagKwise approach as supporting documents are provided.")
            return SSIMRagKwise(
                self.client,
                checkpoint,
                prompt=SSIM_NO_DATA_PROMPT,
                edge_direction_prompt=EDGE_DIRECTION_PROMPT,
                batch=self.batch,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")


    async def _embed_supporting_documents(self, supporting_documents: list[str]):
        store = SimpleVectorStore(name="store", chunks=[])
        for text in supporting_documents:
            try:
                embedding = await self.client.get_embedding(text)
            except Exception as e:
                print(f"Error embedding text: {e}")
                continue
            store.add_chunk(Chunk(
                text=text,
                metadata = {},
                embedding = embedding
            ))
        return store

    def _convert_key_factors(self, factors: list[str]):
        return [Factor(name=f) for f in factors]

    async def get_structural_model(self, research_question: str, factors: list[str], supporting_documents: list[str] = [], checkpoint: str = None):
        vector_store = await self._embed_supporting_documents(supporting_documents)
        approach = self._get_approach(vector_store.chunks, factors, checkpoint)
        if len(vector_store.chunks) == 0:
            return await approach.process_and_save(
                research_question=research_question,
                factors=self._convert_key_factors(factors),
            )
        else:
            return await approach.process_and_save(
                research_question=research_question,
                factors=self._convert_key_factors(factors),
                vector_store=vector_store,
            )