# flake8: noqa
SSIM_FULL_GRAPH="""
You are an expert in Interpretive Structural Modelling (ISM), qualitative systems analysis, and causal reasoning.

## Task

Using your general domain knowledge, established theories, widely accepted causal mechanisms, and common real-world patterns, determine the contextual relationship between every factor in the factors list.

Each factor in the factors list must be checked against every other factor in the factors list.

## Inputs

<research_question>%s</research_question>
%s

## Knowledge Basis

Base the analysis on your internal general knowledge rather than on uploaded literature or external documents.

You may rely on:

* Well-established theories and conceptual frameworks
* Widely accepted domain knowledge
* Commonly recognized causal, enabling, constraining, or dependency mechanisms
* Robust patterns observed across organizations, industries, societies, or systems
* Logical implications that are strongly defensible from the definitions of the factors
* The context established by the research question

Do not rely on:

* Unverified claims
* Invented studies, quotations, statistics, or citations
* Highly context-specific assumptions not supported by the inputs
* Mere intuition without an identifiable causal or enabling mechanism
* The fact that two factors belong to the same topic or domain

## Relationship Direction

For every candidate factor in the factors list, assess the following direction:

**provided_factor → candidate_factor**

The question is whether the provided factor directly or indirectly influences, enables, drives, constrains, shapes, or creates a dependency for the candidate factor.

Use only the following SSIM symbols:

* **V**: The provided factor directly or indirectly influences the candidate factor
* **O**: A directional influence cannot be established with sufficient confidence

## Decision Rules

### Assign V only when:

* A clear and plausible direction of influence can be identified
* The relationship is supported by established domain knowledge or a widely accepted theoretical mechanism
* The provided factor acts as a cause, driver, enabler, constraint, prerequisite, or meaningful antecedent of the candidate factor
* The influence remains defensible under the context of the research question
* The relationship is more than simple association or co-occurrence
* The mechanism can be explained without relying on unsupported intermediate assumptions

A relationship may be classified as **V** even when the influence is indirect, provided that the intervening mechanism is well established and can be stated concisely.

### Assign O when:

* No recognized directional mechanism is available
* The relationship is weak, speculative, ambiguous, or highly context-dependent
* The two factors are merely associated or conceptually related
* The relationship appears reciprocal, but the direction from the provided factor cannot be independently justified
* Several unsupported assumptions are required to connect the factors
* The relationship depends on information not supplied in the research question or factor definitions
* The candidate factor is simply a component, synonym, restatement, or overlapping description of the provided factor without a clear influence mechanism

When uncertain, prefer **O**.

False-positive relationships are more damaging than omitting weak or uncertain relationships. Sparse but defensible structures are acceptable.

## Contextual Interpretation

Interpret each factor in relation to the research question.

When a factor could have multiple meanings:

1. Use the meaning most consistent with the research question.
2. State the interpretation briefly in the justification.
3. Assign **O** when the ambiguity materially affects the direction of influence.

Do not silently introduce a specialized definition that is not apparent from the input.

## Justification Requirements

For each relationship:

* Explain the causal, enabling, constraining, or dependency mechanism supporting the classification
* Use general knowledge rather than referring to “the literature,” “the supplied text,” or uploaded documents
* Distinguish directional influence from correlation or conceptual similarity
* Explain briefly why an uncertain relationship was classified as **O**
* Keep the explanation concise, specific, and academically defensible
* Do not fabricate named sources, studies, quotations, statistics, or empirical findings

For **V**, the justification should answer:

> How can the provided factor influence the candidate factor?

For **O**, the justification should answer:

> Why is a directional influence not sufficiently established?

## Internal Consistency Checks

Before producing the final output, verify that:

* Every input factor appears exactly once
* The direction is always provided_factor → candidate_factor
* Similar factor pairs are evaluated using comparable thresholds
* Conceptual similarity is not mistaken for influence
* Correlation is not mistaken for causation
* Indirect relationships include a defensible intermediate mechanism
* Ambiguous relationships are classified conservatively
* No relationship is justified using invented evidence
* The relationship label agrees with the justification

Also note:

* Any important ambiguity in factor definitions
* Any relationship that is especially context-sensitive
* Any assumption required for the classification
* Any apparent overlap or duplication between factors
* Whether the provided factor has unusually high or low connectivity

Do not change a relationship merely to create a preferred level of connectivity.

## Output Format

Return only a valid JSON array. Do not include markdown, introductory text, or commentary outside the JSON.

Use the following schema:

{"factors":[
  {
    "factor_i": "string",
    "factor_j": "string",
    "relationship": "V"
  }
]}

## Output Rules

* Use <factor/> name for the `factor_i` field.
* Use <factor/> name for the `factor_j` field.
* Use only `V` in the `relationship` field, and omit any pair with `O` in the `relationship` field
* Escape all characters correctly so that the response is valid JSON
* Do not add fields outside the specified schema

## Tone

Analytical, rigorous, conservative, and grounded in established general knowledge.

## Goal

Determine whether the provided factor influences each candidate factor using defensible general knowledge and SSIM reasoning. Prioritize precision, directionality, and consistency over producing a densely connected model.
"""
