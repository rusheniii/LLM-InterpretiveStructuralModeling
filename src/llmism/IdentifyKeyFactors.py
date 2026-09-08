from .LLMClient import append_message
from .prompts.IdentifyKeyFactorsPrompt import KEY_FACTORS_PROMPT
from .model import Factors
from .ISMBaseStep import ISMBaseStep

import logging

from .ssim_utils import document_chunks_to_xml

logger = logging.getLogger(__name__)

class IdentifyKeyFactors(ISMBaseStep):
    async def get_key_factors(
        self,
        research_question: str,
        documents: list[str],
    ) -> Factors:
        messages = []
        xml_documents = document_chunks_to_xml(documents)
        append_message("user", xml_documents, messages)
        user_prompt = (
            "Find the key factors to the research question from the text "
            "provided"
        )
        append_message("user", user_prompt, messages)
        instructions = KEY_FACTORS_PROMPT % research_question.strip()
        return self.client.responses_parse(
            instructions=instructions,
            messages=messages,
            response_format=Factors
        )

    async def process(self, research_question: str, documents: list[str]):
        logger.info("Processing documents: %s", len(documents))
        key_factors = await self.get_key_factors(
            research_question,
            documents,
        )
        logger.info("Identified %s key factors", len(key_factors))
        return key_factors
