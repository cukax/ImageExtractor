"""Architecture tests: the dependency rule is verified, not just documented.

Hexagonal architecture only survives if the import direction is enforced. These
tests parse the source and fail the build the day someone imports boto3 from the
domain.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "src"

#: SDKs that must never be reachable from the inner layers.
PROVIDER_SDKS = frozenset(
    {"openai", "anthropic", "boto3", "botocore", "azure", "google", "httpx"}
)


def _module_level_imports(path: Path) -> list[str]:
    """Return the modules imported at module scope, ignoring function local ones.

    Function local imports are deliberate escape hatches (lazy SDK loading, the
    composition seam in compile_workflow) and are excluded on purpose.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.If):
            # Guarded blocks such as `if TYPE_CHECKING:` never execute.
            continue
    return imported


def _python_files(*relative_paths: str) -> list[Path]:
    """Collect every Python file under the given source subdirectories."""
    files: list[Path] = []
    for relative_path in relative_paths:
        files.extend(sorted((SOURCE_ROOT / relative_path).rglob("*.py")))
    return files


@pytest.mark.parametrize("path", _python_files("domain", "ports"), ids=lambda p: p.name)
def test_the_inner_layers_never_import_infrastructure(path: Path) -> None:
    offenders = [
        module for module in _module_level_imports(path) if "infrastructure" in module
    ]

    assert not offenders, f"{path.name} imports infrastructure: {offenders}"


@pytest.mark.parametrize("path", _python_files("domain", "ports"), ids=lambda p: p.name)
def test_the_inner_layers_never_import_a_provider_sdk(path: Path) -> None:
    offenders = [
        module
        for module in _module_level_imports(path)
        if module.split(".")[0] in PROVIDER_SDKS
    ]

    assert not offenders, f"{path.name} imports a provider SDK: {offenders}"


@pytest.mark.parametrize("path", _python_files("domain"), ids=lambda p: p.name)
def test_the_domain_never_imports_the_orchestrator(path: Path) -> None:
    # The domain must remain usable, and testable, without LangGraph.
    offenders = [
        module for module in _module_level_imports(path) if module.split(".")[0] == "langgraph"
    ]

    assert not offenders, f"{path.name} imports langgraph: {offenders}"


@pytest.mark.parametrize("path", _python_files("graph"), ids=lambda p: p.name)
def test_the_graph_never_imports_infrastructure_at_module_scope(path: Path) -> None:
    offenders = [
        module for module in _module_level_imports(path) if "infrastructure" in module
    ]

    assert not offenders, f"{path.name} imports infrastructure at module scope: {offenders}"


@pytest.mark.parametrize("path", _python_files("graph"), ids=lambda p: p.name)
def test_the_graph_never_imports_a_provider_sdk(path: Path) -> None:
    offenders = [
        module
        for module in _module_level_imports(path)
        if module.split(".")[0] in PROVIDER_SDKS
    ]

    assert not offenders, f"{path.name} imports a provider SDK: {offenders}"


def test_every_adapter_implements_its_port() -> None:
    from src.infrastructure.adapters.extractors import (
        AWSTextractAdapter,
        AzureDocumentIntelligenceAdapter,
        GoogleDocAIAdapter,
        NativeVLMExtractorAdapter,
    )
    from src.infrastructure.adapters.vlm import (
        AnthropicVLMAdapter,
        OllamaVLMAdapter,
        OpenAIVLMAdapter,
    )
    from src.ports.extractor_port import DocumentExtractorPort
    from src.ports.vlm_port import VLMProviderPort

    extractors = (
        AzureDocumentIntelligenceAdapter,
        AWSTextractAdapter,
        GoogleDocAIAdapter,
        NativeVLMExtractorAdapter,
    )
    vlm_providers = (OpenAIVLMAdapter, AnthropicVLMAdapter, OllamaVLMAdapter)

    for adapter in extractors:
        assert issubclass(adapter, DocumentExtractorPort)
        assert adapter.provider_name != "unknown"
    for adapter in vlm_providers:
        assert issubclass(adapter, VLMProviderPort)


def test_the_ports_cannot_be_instantiated_directly() -> None:
    from src.ports.extractor_port import DocumentExtractorPort
    from src.ports.vlm_port import VLMProviderPort

    with pytest.raises(TypeError):
        DocumentExtractorPort()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        VLMProviderPort()  # type: ignore[abstract]
