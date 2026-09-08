# LLM-ISM

> Interpretive Structural Modeling with large language models.

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2E7D32)](https://opensource.org/license/mit/)

## 1. What is LLM-ISM?

Interpretive Structural Modeling (ISM) usually relies on repeated consultation with subject-matter experts to establish causal relationships between factors. That process is valuable, but labor-intensive and difficult to scale.

LLM-ISM uses large language models to discover and organize those relationships into a structural model. It can work from a research question and factor list alone, or ground its decisions in a collection of supporting research papers.

The accompanying study evaluates pairwise, k-wise, rowwise, and full-graph strategies. In that evaluation, the full-graph approach achieved the strongest results (SHD=0.27, F1-score=0.73).


## 2. Features

- **LLM-driven structural discovery**: infer directed relationships between research factors.
- **Multiple discovery strategies**: use `pairwise`,`k-wise`,`rowwise` or `full` processing for factor comparisons.
- **Retrieval-augmented analysis**: extract and embed supporting PDFs before relationship inference.
- **Resumable workflows**: write checkpoints as processing completes.
- **Reusable Python API**: integrate structural modeling into notebooks, scripts, and larger research pipelines.
- **Included study data**: explore examples covering circular economy, fruit and vegetables, generative AI in higher education, Industry 4.0, social media, and truck-driver shortage research.

## 3. Coming Soon
- More evaluation datasets and comparison metrics.
- A richer visualization workflow for inspecting discovered structural models.
- Expanded documentation for custom prompts and retrieval pipelines.

## 4. Quick Start

### 4.1 via Docker-Compose + Jupyter Notebook

An example jupyter notebook is provided in the example folder. Docker Compose will start a jupyter notebook server with all the required libraries. 

```bash
git clone https://github.com/rusheniii/LLM-InterpretiveStructuralModeling.git
cd LLM-InterpretiveStructuralModeling
docker-compose up
```

Copy the url from the logs `http://localhost:8000/tree?token=<token>`. 

### 4.2 Python Install
LLM-ISM requires Python 3.13 or newer and an OpenAI-compatible asynchronous API endpoint.

```bash
git clone https://github.com/rusheniii/LLM-InterpretiveStructuralModeling.git
cd LLM-InterpretiveStructuralModeling

python -m venv .venv
source .venv/bin/activate
pip install .
```

For PDF extraction support, install the optional extra:

```bash
pip install "llmism[extraction]"
```


## 5. Example
The library accepts an `AsyncOpenAI` client, a chat model, an embedding model, and an ISM approach. This example runs a full-graph study without supporting documents:

```python
import asyncio
import os

from openai import AsyncOpenAI

from llmism import Approach, InterpretiveStructuralModeling


async def discover_structure():
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    ism = InterpretiveStructuralModeling(
        client=client,
        model="gpt-4o",
        embedding_model="text-embedding-3-large",
        method=Approach.FULL,
    )

    graph = await ism.get_structural_model(
        research_question="What contributes to the adoption of electric vehicles?",
        factors=[
            "Charging infrastructure",
            "Vehicle cost",
            "Consumer awareness",
        ],
    )
    return graph


graph = asyncio.run(discover_structure())
print(graph.nodes)
```

For an OpenAI-compatible service hosted elsewhere, pass its endpoint with the client's `base_url` argument.

## 6. References

If you use this repository, please cite:

```bibtex
@inproceedings{rush_overcoming_ism_llms,
	title = {Overcoming Challenges of Interpretive Structural Modeling with Large Language Models},
	author = {Rush, Everett and Icove, David J. and Kim, Ari and Park, Byung H. and Langston, Michael A.},
	affiliation = {Department of Electrical Engineering and Computer Science, University of Tennessee; Oak Ridge National Laboratory},
}
```

## 7. License

LLM-ISM is released under the [MIT License](https://opensource.org/license/mit/). See the package metadata in `pyproject.toml` for the project license declaration.
