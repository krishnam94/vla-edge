"""CLI entry point for vla-edge."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="vla-edge",
    help="Profile, optimize, validate, and deploy VLA models to edge hardware.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def check() -> None:
    """Show hardware info and backend availability."""
    from vla_edge.registry import list_backends

    table = Table(title="vla-edge - Hardware Check")
    table.add_column("Backend", style="cyan")
    table.add_column("Available", style="green")

    backends = list_backends()
    for name, available in backends.items():
        status = "[green]YES[/green]" if available else "[red]NO[/red]"
        table.add_row(name, status)

    console.print(table)

    # Show active backend details
    from vla_edge.registry import get_backend

    try:
        backend = get_backend("auto")
        caps = backend.get_capabilities()
        console.print(f"\n[bold]Active backend:[/bold] {caps.name}")
        console.print(f"  Memory: {caps.memory_mb} MB")
        console.print(f"  Supported dtypes: {', '.join(caps.supported_dtypes)}")
        console.print(f"  Supported formats: {', '.join(caps.supported_formats)}")
        if caps.cuda_version:
            console.print(f"  CUDA: {caps.cuda_version}")
        if caps.jetpack_version:
            console.print(f"  JetPack: {caps.jetpack_version}")
    except RuntimeError:
        console.print("\n[red]No backend available.[/red]")


@app.command()
def profile(
    model: str = typer.Argument(..., help="Model name from registry or HuggingFace ID"),
    hardware: str = typer.Option("auto", help="Backend: auto, cpu, cuda, jetson"),
    iterations: int = typer.Option(100, help="Number of inference iterations"),
    warmup: int = typer.Option(10, help="Warmup iterations before timing"),
    output: Path | None = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """Profile a VLA model's latency, memory, and throughput."""
    from vla_edge.profile.latency import run_profile

    console.print(f"[bold]Profiling[/bold] {model} on {hardware} ({iterations} iterations)...")
    result = run_profile(
        model_name=model,
        backend_name=hardware,
        iterations=iterations,
        warmup=warmup,
    )

    # Display results
    table = Table(title=f"Profile: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Backend", result.get("backend", "unknown"))
    table.add_row("Avg latency", f"{result['avg_ms']:.1f} ms")
    table.add_row("P95 latency", f"{result['p95_ms']:.1f} ms")
    table.add_row("P99 latency", f"{result['p99_ms']:.1f} ms")
    table.add_row("FPS", f"{result['fps']:.1f}")
    table.add_row("Peak memory", f"{result['peak_memory_mb']:.1f} MB")

    console.print(table)

    if output:
        import json

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        console.print(f"\nResults saved to {output}")


@app.command()
def models() -> None:
    """List available VLA model adapters."""
    from vla_edge.registry import list_models

    available = list_models()
    if not available:
        console.print("[yellow]No models registered. Install model extras or add custom models.[/yellow]")
        return

    for name in available:
        console.print(f"  - {name}")


@app.command()
def version() -> None:
    """Show vla-edge version."""
    from vla_edge import __version__

    console.print(f"vla-edge {__version__}")


if __name__ == "__main__":
    app()
