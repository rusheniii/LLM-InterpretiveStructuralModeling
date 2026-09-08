from .model import Factor

def document_chunks_to_xml(chunks: list[str]) -> str:
    documents_xml = ''.join([document_chunk_to_xml(chunk) for chunk in chunks])
    return f"<documents>{documents_xml}</documents>"

def document_chunk_to_xml(chunk: str) -> str:
    return f"<document>{chunk}</document>"

def factors_to_xml(factors: list[Factor]) -> str:
    return factor_names_to_xml([f.name for f in factors])

def factor_names_to_xml(factors: list[str]) -> str:
    factors_xml = "".join([factor_to_xml(f) for f in factors])
    return f"<factors>{factors_xml}</factors>"

def factor_to_xml(factor: str) -> str:
    return f"<factor><factor_name>{factor}</factor_name></factor>"

def provided_factor_to_xml(factor: Factor) -> str:
    return (
        "<provided_factor>"
        f"<provided_factor_name>{factor.name}</provided_factor_name>"
        "</provided_factor>"
    )
