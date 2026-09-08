FROM quay.io/jupyter/minimal-notebook

USER root
RUN apt update
RUN apt install -y build-essential ca-certificates libssl-dev wget

USER jovyan
WORKDIR /home/jovyan
COPY README.md .
COPY --chown=jovyan:jovyan src/ ./src/
COPY --chown=jovyan:jovyan example ./example
COPY pyproject.toml .
RUN pip install .
RUN pip install matplotlib pygraphviz
EXPOSE 8000
CMD ["jupyter", "notebook","--port","8000"]
