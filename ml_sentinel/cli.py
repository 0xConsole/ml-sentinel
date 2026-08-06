#!/usr/bin/env python3
"""ML Sentinel CLI — the demo entry point.

Usage:
    # Mock demo (no DataHub needed — uses the bundled sample estate):
    ml-sentinel scan --mock
    ml-sentinel scan --mock --json          # machine-readable output
    ml-sentinel scan --mock --no-write       # don't write findings back (dry run)
    ml-sentinel lineage --mock              # print the lineage graph
    ml-sentinel estate --mock               # list all entities in the estate

    # Against a real DataHub via MCP:
    ml-sentinel scan --mcp --datahub-url http://localhost:8080 --token ...

    # Against a real DataHub via GraphQL / SDK:
    ml-sentinel scan --graphql --datahub-url http://localhost:8080 --token ...
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .agent import MLSentinelAgent
from .datahub_client import (
    GraphQLDataHubClient,
    MCPDataHubClient,
    MockDataHubClient,
)
from .models import EntityType


def build_client(args) -> object:
    """Construct the appropriate DataHub client from CLI args.

    Defaults to mock mode so the demo always works out of the box.
    """
    if getattr(args, "mock", False) or not (
        getattr(args, "mcp", False) or getattr(args, "graphql", False)
    ):
        return MockDataHubClient()
    if getattr(args, "graphql", False):
        return GraphQLDataHubClient(
            server=args.datahub_url, token=args.token
        )
    if getattr(args, "mcp", False):
        return MCPDataHubClient(
            datahub_url=args.datahub_url, datahub_token=args.token
        )
    return MockDataHubClient()


def cmd_scan(args) -> int:
    client = build_client(args)
    agent = MLSentinelAgent(client, write_back=not args.no_write)
    result = agent.scan()

    if args.json:
        _print_json(result, client)
    else:
        _print_rich(result, client)
    return 1 if result.critical_count > 0 else 0


def cmd_lineage(args) -> int:
    client = build_client(args)
    if not isinstance(client, MockDataHubClient):
        print("Lineage view is only available in mock mode (the live transport "
              "requires a DataHub instance).")
        return 1
    _print_lineage(client)
    return 0


def cmd_estate(args) -> int:
    client = build_client(args)
    if not isinstance(client, MockDataHubClient):
        print("Estate view is only available in mock mode.")
        return 1
    _print_estate(client)
    return 0


def _print_rich(result, client) -> None:
    """Pretty-print the scan result using `rich`."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        _print_plain(result, client)
        return

    console = Console()

    # Header
    console.print(
        Panel.fit(
            Text("ML Sentinel — Production ML Silent-Problem Hunter", style="bold cyan"),
            subtitle="Powered by DataHub lineage",
        )
    )

    # Summary
    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Entities scanned", str(result.entities_scanned))
    summary.add_row("Findings", str(len(result.findings)))
    summary.add_row(
        "Severity",
        f"[red]{result.critical_count} critical[/red] · "
        f"[yellow]{result.warn_count} warn[/yellow] · "
        f"[blue]{result.info_count} info[/blue]",
    )
    summary.add_row("Duration", f"{result.duration_seconds:.2f}s")
    if result.assertions_written or result.tags_written:
        summary.add_row(
            "Written back to DataHub",
            f"{result.assertions_written} assertions · "
            f"{result.tags_written} tags · "
            f"{result.documents_written} documents",
        )
    console.print(summary)

    # Findings table
    if result.findings:
        console.print()
        table = Table(
            title="Silent Problems Detected",
            show_lines=True,
            header_style="bold",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Severity", width=10)
        table.add_column("Type", width=22)
        table.add_column("Entity", width=28)
        table.add_column("Finding", overflow="fold", ratio=1)
        for i, f in enumerate(result.findings, 1):
            sev = f.severity.value
            color = {"CRITICAL": "red", "WARN": "yellow", "INFO": "blue"}.get(sev, "white")
            table.add_row(
                str(i),
                f"[{color}]{sev}[/{color}]",
                f.type.value,
                f.evidence.get("entity_name", f.entity_urn),
                f.title,
            )
        console.print(table)

        # Evidence detail
        console.print()
        for i, f in enumerate(result.findings, 1):
            sev = f.severity.value
            color = {"CRITICAL": "red", "WARN": "yellow", "INFO": "blue"}.get(sev, "white")
            console.print(
                Panel(
                    f"[bold]{f.title}[/bold]\n\n"
                    f"{f.description}\n\n"
                    f"[dim]Evidence:[/dim]",
                    title=f"[{color}]#{i} {sev}[/{color}] · {f.type.value}",
                    border_style=color,
                )
            )
            ev_table = Table(show_header=False, box=None, padding=(0, 2))
            ev_table.add_column(style="dim")
            ev_table.add_column()
            for k, v in f.evidence.items():
                ev_table.add_row(str(k), str(v))
            console.print(ev_table)
            console.print(f"\n[dim]Remediation:[/dim] {f.remediation}")
            console.print(f"[dim]Assertion URN:[/dim] {f.assertion_urn}")
            console.print(f"[dim]Tag:[/dim] {f.tag_name}")
            console.print()
    else:
        console.print("\n[green]✓ No silent problems detected. The ML estate is healthy.[/green]")

    # Write-back summary (mock mode)
    if isinstance(client, MockDataHubClient) and (result.assertions_written or result.tags_written):
        console.print(
            Panel(
                f"[green]Wrote back to DataHub:[/green]\n"
                f"  • {len(client.written_assertions)} assertions created\n"
                f"  • {sum(len(t) for t in client.written_tags.values())} tags applied\n"
                f"  • {len(client.written_documents)} documents saved\n\n"
                f"[dim]In mock mode these are recorded in-memory. With a live "
                f"DataHub they become real catalog objects visible to every "
                f"user and agent.[/dim]",
                title="Write-Back",
                border_style="green",
            )
        )


def _print_plain(result, client) -> None:
    """Fallback plain-text output when rich isn't installed."""
    print(f"\nML Sentinel — scan complete")
    print(f"  Entities scanned: {result.entities_scanned}")
    print(f"  Findings: {len(result.findings)} "
          f"({result.critical_count} critical, {result.warn_count} warn, "
          f"{result.info_count} info)")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    if result.assertions_written:
        print(f"  Written back: {result.assertions_written} assertions, "
              f"{result.tags_written} tags, {result.documents_written} documents")
    print()
    for i, f in enumerate(result.findings, 1):
        print(f"  [{f.severity.value}] #{i} {f.type.value} — {f.title}")
        print(f"    Entity: {f.evidence.get('entity_name', f.entity_urn)}")
        print(f"    {f.description}")
        for k, v in f.evidence.items():
            print(f"    {k}: {v}")
        print(f"    Remediation: {f.remediation}")
        print(f"    Assertion: {f.assertion_urn}")
        print(f"    Tag: {f.tag_name}")
        print()


def _print_json(result, client) -> None:
    payload = {
        "entities_scanned": result.entities_scanned,
        "findings_count": len(result.findings),
        "critical": result.critical_count,
        "warn": result.warn_count,
        "info": result.info_count,
        "by_type": result.by_type(),
        "duration_seconds": round(result.duration_seconds, 3),
        "assertions_written": result.assertions_written,
        "tags_written": result.tags_written,
        "documents_written": result.documents_written,
        "findings": [f.to_dict() for f in result.findings],
    }
    if isinstance(client, MockDataHubClient):
        payload["written_back"] = {
            "assertions": [f.to_dict() for f in client.written_assertions],
            "tags": {urn: list(tags) for urn, tags in client.written_tags.items()},
            "documents": client.written_documents,
        }
    print(json.dumps(payload, indent=2, default=str))


def _print_lineage(client: MockDataHubClient) -> None:
    try:
        from rich.console import Console
        from rich.tree import Tree
        from rich.text import Text
    except ImportError:
        print("Install `rich` for the lineage view: pip install rich")
        return

    console = Console()
    estate = client.estate
    tree = Tree(Text("ML Lineage Graph (fraud-detection pipeline)", style="bold cyan"))

    # Build a simple parent -> children map
    children: dict[str, list[str]] = {}
    entity_map = {e.urn: e for e in estate.entities}
    for edge in estate.edges:
        children.setdefault(edge.upstream_urn, []).append(edge.downstream_urn)

    visited: set[str] = set()

    def add(node_urn: str, parent) -> None:
        if node_urn in visited:
            return
        visited.add(node_urn)
        entity = entity_map.get(node_urn)
        label = entity.name if entity else node_urn
        etype = entity.type.value if entity else "?"
        node = parent.add(Text(f"{label}  ({etype})", style="bold"))
        for child in children.get(node_urn, []):
            add(child, node)

    # Start from roots (entities with no upstream)
    all_downstream = {e.downstream_urn for e in estate.edges}
    roots = [e.urn for e in estate.entities if e.urn not in all_downstream]
    for root in roots:
        add(root, tree)

    console.print(tree)
    console.print(
        "\n[dim]Note: the edge raw_transactions → user_txn_features is "
        "intentionally absent (broken lineage).[/dim]"
    )


def _print_estate(client: MockDataHubClient) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        for e in client.estate.entities:
            print(f"{e.type.value:25s}  {e.name}")
        return

    console = Console()
    table = Table(title="DataHub ML Estate (mock)", show_lines=True)
    table.add_column("Type", style="bold")
    table.add_column("Name")
    table.add_column("URN", overflow="fold")
    table.add_column("Last Modified")
    for e in client.estate.entities:
        table.add_row(
            e.type.value,
            e.name,
            e.urn,
            e.last_modified.isoformat() if e.last_modified else "-",
        )
    console.print(table)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the transport + option flags to a parser (shared across subcommands)."""
    transport = p.add_mutually_exclusive_group()
    transport.add_argument("--mock", action="store_true", default=False,
                           help="Use the bundled mock estate (default, no DataHub needed)")
    transport.add_argument("--mcp", action="store_true", default=False,
                           help="Connect to DataHub via the MCP server")
    transport.add_argument("--graphql", action="store_true", default=False,
                           help="Connect to DataHub via the Python SDK / GraphQL")
    p.add_argument("--datahub-url", default="http://localhost:8080",
                   help="DataHub server URL (for --mcp / --graphql)")
    p.add_argument("--token", default=None, help="DataHub personal access token")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--no-write", action="store_true",
                   help="Don't write findings back to DataHub (dry run)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ml-sentinel",
        description="ML Sentinel — Production ML silent-problem hunter for DataHub.",
    )
    sub = parser.add_subparsers(dest="command")

    sub_scan = sub.add_parser("scan", help="Run the full detector fleet (default)")
    sub_scan = _augment_parser(sub_scan)

    sub_lineage = sub.add_parser("lineage", help="Print the lineage graph (mock)")
    sub_lineage = _augment_parser(sub_lineage)

    sub_estate = sub.add_parser("estate", help="List all entities (mock)")
    sub_estate = _augment_parser(sub_estate)

    args = parser.parse_args(argv)
    command = args.command or "scan"

    # If no subcommand was given, parse with common args on the top-level parser.
    if not args.command:
        # Re-parse so flags like --mock / --json work without a subcommand.
        parser2 = argparse.ArgumentParser(prog="ml-sentinel")
        _add_common_args(parser2)
        args = parser2.parse_args(argv)
        args.command = "scan"

    if command == "scan":
        return cmd_scan(args)
    elif command == "lineage":
        return cmd_lineage(args)
    elif command == "estate":
        return cmd_estate(args)
    else:
        parser.print_help()
        return 0


def _augment_parser(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common args to a subparser and return it."""
    _add_common_args(p)
    return p


if __name__ == "__main__":
    sys.exit(main())
