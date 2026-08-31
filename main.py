"""Command line entry point for the ImageExtractor workflows.

Two modes, sharing one composition root:

* Single document, human-in-the-loop. Escalates to an operator and can be resumed
  from a later invocation.
* Dossier, headless. Splits a multi-document PDF, processes every document in
  parallel and always returns a terminal report.

Examples:
    python main.py --image ./samples/ine_front.jpg --doc-type INE
    python main.py --image ./samples/invoice.png --doc-type Invoice --thread-id inv-001
    python main.py --resume inv-001 --approve --correct total_amount=1432.09
    python main.py --dossier ./samples/kyc.pdf --dossier-id DOSSIER_2026_99482
"""

from __future__ import annotations

import truststore

# Make every ssl.SSLContext (httpx, openai, anthropic, boto3, google-api-core)
# verify against the OS certificate store instead of the bundled certifi CAs.
# This must run before any HTTP client is constructed: on machines where a
# corporate proxy or endpoint security agent performs TLS interception, the
# interception root CA is trusted by Windows but absent from certifi, which
# otherwise fails every outbound call with SSLCertVerificationError.
truststore.inject_into_ssl()

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from src.domain.orchestrator_state import DossierStatus
from src.domain.state import DocumentStatus
from src.graph.orchestrator import compile_dossier_workflow, run_dossier
from src.graph.workflow import compile_workflow, resume_document, run_document
from src.infrastructure.config import build_dependencies, get_settings
from src.ports.extractor_port import ExtractionError
from src.ports.pdf_port import PDFProcessingError
from src.ports.vlm_port import VLMProviderError

LOGGER = logging.getLogger("ImageExtractor")

# Keys that must never be printed: they hold raw image payloads.
_BINARY_STATE_KEYS = frozenset({"image_bytes", "original_image_bytes"})

# Suspended runs must survive between separate CLI invocations (--image today,
# --resume in a later process), so the CLI needs a checkpointer backed by a
# file rather than compile_workflow()'s in-memory default, which is emptied
# the moment this process exits.
CHECKPOINT_DB_PATH = Path(__file__).resolve().parent / "checkpoints.sqlite3"


def _configure_logging(level: str) -> None:
    """Configure root logging with a format suited to container logs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _printable_state(state: dict[str, Any]) -> dict[str, Any]:
    """Strip binary payloads so the state can be serialized to JSON."""
    printable: dict[str, Any] = {}
    for key, value in state.items():
        if key in _BINARY_STATE_KEYS:
            printable[key] = f"<{len(value or b'')} bytes>"
        else:
            printable[key] = value
    return printable


def _parse_corrections(raw_corrections: list[str]) -> dict[str, str]:
    """Parse repeated --correct field=value arguments."""
    corrections: dict[str, str] = {}
    for item in raw_corrections:
        if "=" not in item:
            raise SystemExit(f"Invalid correction {item!r}; expected the form field=value.")
        field_name, value = item.split("=", 1)
        corrections[field_name.strip()] = value.strip()
    return corrections


def _has_suspended_run(compiled_workflow: Any, thread_id: str) -> bool:
    """True when the checkpointer actually holds a suspended run for this thread.

    Resuming an unknown thread does not raise: LangGraph just starts a fresh
    run from START with an empty state, which then fails deep inside the graph
    with a confusing error instead of a clear "nothing to resume" message.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = compiled_workflow.get_state(config)
    return bool(snapshot.values)


def _report_interrupt(compiled_workflow: Any, thread_id: str) -> bool:
    """Print the pending human review payload, if the run is suspended.

    Returns True when the workflow is waiting for an operator.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = compiled_workflow.get_state(config)
    pending = [
        task_interrupt
        for task in getattr(snapshot, "tasks", []) or []
        for task_interrupt in getattr(task, "interrupts", []) or []
    ]
    if not pending:
        return False

    print("\n=== HUMAN REVIEW REQUIRED ===")
    print(json.dumps(pending[0].value, indent=2, ensure_ascii=False, default=str))
    print(
        f"\nResume with:\n"
        f"  python main.py --resume {thread_id} --approve "
        f"--correct <field>=<value>"
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    """Declare the command line interface."""
    parser = argparse.ArgumentParser(description="Run the ImageExtractor workflows.")
    parser.add_argument("--image", type=Path, help="Path to a single document image.")
    parser.add_argument(
        "--dossier",
        type=Path,
        help="Path to a multi-document PDF, processed headlessly in parallel.",
    )
    parser.add_argument(
        "--dossier-id",
        default=None,
        help="Identifier prefixed to every doc_id; generated when omitted.",
    )
    parser.add_argument(
        "--doc-type",
        default="INE",
        help="Document type: INE, Invoice, Form, ... (single document mode only).",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Checkpointer thread id; generated when omitted.",
    )
    parser.add_argument("--resume", metavar="THREAD_ID", help="Resume a suspended run.")
    parser.add_argument("--approve", action="store_true", help="Approve on resume.")
    parser.add_argument(
        "--correct",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Field correction applied on resume; repeatable.",
    )
    return parser


def _run_dossier_mode(dependencies: Any, dossier_path: Path, dossier_id: str | None) -> int:
    """Run a dossier end to end and print the master JSON contract.

    No checkpointer is passed: the dossier engine is headless by construction and
    never suspends, so there is nothing to resume and nothing to persist between
    invocations.
    """
    if not dossier_path.is_file():
        raise SystemExit(f"Dossier not found: {dossier_path}")

    workflow = compile_dossier_workflow(dependencies)
    report = run_dossier(
        workflow,
        pdf_bytes=dossier_path.read_bytes(),
        dossier_id=dossier_id,
    )

    print(json.dumps(report.to_json_dict(), indent=2, ensure_ascii=False, default=str))
    return 0 if report.dossier_status == DossierStatus.COMPLETED else 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    _configure_logging(settings.log_level)

    if not args.image and not args.resume and not args.dossier:
        build_parser().error("One of --image, --dossier or --resume is required.")

    try:
        dependencies = build_dependencies(settings)
    except (ExtractionError, PDFProcessingError, VLMProviderError) as error:
        # A missing credential is a configuration problem, not a crash: report
        # it as one line instead of a stack trace.
        raise SystemExit(f"Configuration error: {error}") from error

    try:
        if args.dossier:
            return _run_dossier_mode(dependencies, args.dossier, args.dossier_id)

        with SqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
            workflow = compile_workflow(dependencies, checkpointer=checkpointer)

            if args.resume:
                thread_id = args.resume
                if not _has_suspended_run(workflow, thread_id):
                    raise SystemExit(
                        f"No suspended run found for thread {thread_id!r} in "
                        f"{CHECKPOINT_DB_PATH.name}. It may already be finished, the "
                        f"thread id may be mistyped, or it was created against a "
                        f"different checkpoint database."
                    )
                final_state = resume_document(
                    workflow,
                    thread_id=thread_id,
                    approved=args.approve,
                    corrections=_parse_corrections(args.correct),
                )
            else:
                image_path: Path = args.image
                if not image_path.is_file():
                    raise SystemExit(f"Image not found: {image_path}")
                thread_id = args.thread_id or f"doc-{uuid.uuid4().hex[:12]}"
                final_state = run_document(
                    workflow,
                    image_bytes=image_path.read_bytes(),
                    doc_type=args.doc_type,
                    thread_id=thread_id,
                )

            if _report_interrupt(workflow, thread_id):
                return 2

            print(json.dumps(_printable_state(final_state), indent=2, ensure_ascii=False, default=str))
            status = final_state.get("status")
            return 0 if status == DocumentStatus.VALIDATED else 1
    finally:
        dependencies.close()


if __name__ == "__main__":
    sys.exit(main())
