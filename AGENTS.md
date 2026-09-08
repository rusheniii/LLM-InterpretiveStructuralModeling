# Repository Instructions for Codex Agents

## Purpose
- Use this file as the default implementation workflow for feature work in this repository.
- Do not consider a feature complete until code changes, unit tests, and a Docker-based integration check have all been done.

## Repository Map

### Backend
- Path: `src/`
- Stack: Python
- Main app entry: `src/main.py`
- Prompts: `src/prompts/`
- Logic: `src/` 
- Model: `src/model.py`
- Tests: `tests/`
- Pytest config: `pytest.ini`


## Feature Workflow Requirements
- Start by reading the relevant entrypoints before editing.
- Keep changes localized to the feature area
- Every feature must include unit tests.
- Do not finish with only static analysis or code edits; run the required checks.

## Backend Feature Rules
- Add or update unit tests in `tests/`.
- If you add dependencies, document them in the change and keep setup minimal.
- each test file should have a test class that inherits from IsolatedAsyncioTestCase or TestCase. 
- Use AsyncMock and Mock for dependencies. 
- Each new test should follow this format: 
    - expected response object
    - mock setup
    - method invocation
    - assertions
- Each new test should cover exactly one method in the callee class. 
- Assert that the correct methods are called on all mocks with the expected object.
- Use the existing tests as an example of Mocking and assertion.
- Dont stop until you get to 99% line coverage for a specific feature.


### Backend unit test commands
- Run backend tests:

```bash
source .venv/bin/activate
pytest
```


## Required Integration Check for Every Feature
- Run the relevant unit tests first.

## Completion Checklist
- Code implemented
- Unit tests added or updated
- Backend tests run for backend changes
 