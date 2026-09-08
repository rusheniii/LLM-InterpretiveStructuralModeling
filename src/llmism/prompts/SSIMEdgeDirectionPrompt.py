# flake8: noqa
EDGE_DIRECTION_PROMPT = """
You are an expert in Interpretive Structural Modelling (ISM) and qualitative systems analysis.

# Task

A previous analysis has already established that a causal or influential relationship exists between the provided factor and every factor in the factor list.

Your task is to determine **the direction of influence** between the provided factor and each factor using the supplied supporting literature.

## Inputs

<research_question>%s</research_question>
%s
%s

Supporting Literature:
{Text uploaded in this session}

---

# Instructions

## Pairwise Direction Analysis

For each factor in the factors list, compare it with the provided factor.

Determine the direction of influence using the following SSIM symbols.

**V**
The provided factor influences the comparison factor.

**A**
The comparison factor influences the provided factor.

**O**
A directional influence cannot be established with sufficient confidence

---

## Determining Direction

Infer direction from the supporting literature by identifying which factor functions as the cause, driver, prerequisite, enabler, antecedent, constraint, or mechanism, and which factor functions as the consequence, outcome, dependency, or effect.

Prefer directionality supported by:

- Explicit causal statements.
- Dependency relationships.
- Enabling mechanisms.
- Established theoretical frameworks.
- Widely accepted domain knowledge when consistent with the literature.

Consider both direct and indirect influence.

---

# Output Format

Return only a valid JSON array. Do not include markdown, introductory text, or commentary outside the JSON.

Use the following schema:

{"factors":[
  {
    "provided_factor_name": "string",
    "factor_name": "string",
    "relationship": "V or A",
  }
]}

---

## Constraints

* Preserve each factor's supplied identifier and name
* Produce one object for every factor in the factors list
* Keep the original order of the factors
* Use only 'V' or 'A' in the 'relationship' field, and omit any pair with 'O' in the 'relationship' field
* Escape all characters correctly so that the response is valid JSON
* Do not add fields outside the specified schema

---

## Goal

Determine whether the provided factor influences each candidate factor using defensible general knowledge and SSIM reasoning. Prioritize precision, directionality, and consistency over producing a densely connected model.
"""
