"""Command line entry point for the ImageExtractor workflow.

Examples:
    python main.py --image ./samples/ine_front.jpg --doc-type INE
    python main.py --image ./samples/invoice.png --doc-type Invoice --thread-id inv-001
    python main.py --resume inv-001 --approve --correct total_amount=1432.09
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from src.domain.state import DocumentStatus
from src.graph.workflow import compile_workflow, resume_document, run_document
from src.infrastructure.config import build_dependencies, get_settings
from src.ports.extractor_port import ExtractionError
from src.ports.vlm_port import VLMProviderError

LOGGER = logging.getLogger("ImageExtractor")

# Keys that must never be printed: they hold raw image payloads.
_BINARY_STATE_KEYS = frozenset({"image_bytes", "original_image_bytes"})


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
    parser = argparse.ArgumentParser(description="Run the ImageExtractor workflow.")
    parser.add_argument("--image", type=Path, help="Path to the document image.")
    parser.add_argument(
        "--doc-type",
        default="INE",
        help="Document type: INE, Invoice, Form, ...",
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


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    _configure_logging(settings.log_level)

    if not args.image and not args.resume:
        build_parser().error("Either --image or --resume is required.")

    try:
        dependencies = build_dependencies(settings)
    except (ExtractionError, VLMProviderError) as error:
        # A missing credential is a configuration problem, not a crash: report
        # it as one line instead of a stack trace.
        raise SystemExit(f"Configuration error: {error}") from error

    try:
        # NOTE: the default in memory checkpointer only lives as long as this
        # process, so --resume works within a session. Wire a durable
        # checkpointer to resume across restarts.
        workflow = compile_workflow(dependencies)

        if args.resume:
            thread_id = args.resume
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
