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
    table.add_row("Image size", str(result.get("image_size", "unknown")))
    cold = result.get("cold_start_ms", 0)
    if cold > 1000:
        table.add_row("[bold]Cold start[/bold]", f"[red]{cold/1000:.1f} s[/red]")
    else:
        table.add_row("Cold start", f"{cold:.1f} ms")
    table.add_row("Cached avg", f"{result['avg_ms']:.1f} ms")
    table.add_row("Stddev", f"{result.get('stddev_ms', 0):.1f} ms")
    table.add_row("P95", f"{result['p95_ms']:.1f} ms")
    table.add_row("P99", f"{result['p99_ms']:.1f} ms")
    table.add_row("FPS (cached)", f"{result.get('fps_cached', 0):.1f}")
    table.add_row("Peak memory", f"{result['peak_memory_mb']:.1f} MB")
    if result.get("uses_action_chunking"):
        table.add_row("[yellow]Note[/yellow]", result.get("chunking_note", "Uses action chunking"))
    if result.get("warning"):
        table.add_row("[yellow]Warning[/yellow]", result["warning"])
    table.add_row("Safety", "[yellow]Run 'vla-edge validate' for safety analysis[/yellow]")

    console.print(table)

    if output:
        import json

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        console.print(f"\nResults saved to {output}")


@app.command()
def validate(
    model: str = typer.Argument(..., help="Model name from registry or HuggingFace ID"),
    hardware: str = typer.Option("auto", help="Backend: auto, cpu, cuda, jetson"),
    iterations: int = typer.Option(10, help="Number of inference iterations to validate"),
    action_bound: float = typer.Option(1.0, help="Symmetric action bound [-bound, bound]"),
    velocity_limit: float = typer.Option(0.1, help="Max action change per step"),
    output: Path | None = typer.Option(None, help="Save results to JSON file"),
) -> None:
    """Validate a model's action safety (bounds, velocity, workspace)."""
    import json

    import numpy as np

    from vla_edge.profile.latency import _create_dummy_observation, run_profile
    from vla_edge.validate.guard import SafetyGuard
    from vla_edge.validate.safety import SafetyConfig

    console.print(f"[bold]Validating safety[/bold] for {model} ({iterations} iterations)...")

    # Build safety config
    config = SafetyConfig(
        action_bounds=np.array([[-action_bound, action_bound]] * 7, dtype=np.float32),
        max_velocity=np.full(7, velocity_limit, dtype=np.float32),
    )

    # Run profile which also loads the model
    profile_result = run_profile(
        model_name=model,
        backend_name=hardware,
        iterations=iterations,
        warmup=2,
    )

    # Collect actions from a separate run for safety analysis
    from vla_edge.registry import get_backend, get_model

    backend = get_backend(hardware)
    try:
        model_cls = get_model(model)
        device = "cuda" if hardware in ("cuda", "jetson") else "cpu"
        loaded_model = model_cls(device=device)
    except (KeyError, TypeError):
        loaded_model = backend.load_model(model, trust_remote_code=True)

    guard = SafetyGuard(config)
    obs = _create_dummy_observation(seed=42)

    for _ in range(iterations):
        guard.safe_infer(backend, loaded_model, obs)

    # Display results
    table = Table(title=f"Safety Validation: {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    summary = guard.summary
    table.add_row("Total calls", str(summary["total_calls"]))
    table.add_row("Calls with violations", str(summary["calls_with_violations"]))
    table.add_row(
        "Violation rate",
        f"[red]{summary['violation_rate']:.1%}[/red]"
        if summary["violation_rate"] > 0
        else f"[green]{summary['violation_rate']:.1%}[/green]",
    )
    table.add_row("Critical violations", str(summary["critical_violations"]))
    table.add_row("Max velocity observed", f"{summary['max_velocity_observed']:.4f}")
    table.add_row("Avg latency", f"{profile_result['avg_ms']:.1f} ms")
    table.add_row("FPS", f"{profile_result['fps']:.1f}")

    console.print(table)

    is_safe = summary["critical_violations"] == 0
    if is_safe:
        console.print("\n[green bold]PASS[/green bold] - No critical safety violations")
    else:
        console.print("\n[red bold]FAIL[/red bold] - Critical safety violations detected")

    if output:
        combined = {**profile_result, "safety": summary, "safety_pass": is_safe}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(combined, indent=2))
        console.print(f"Results saved to {output}")

    # Cleanup
    if hasattr(loaded_model, "cleanup"):
        loaded_model.cleanup()


@app.command()
def optimize(
    model: str = typer.Argument(..., help="Model name from registry or HuggingFace ID"),
    format: str = typer.Option("onnx", help="Export format: onnx"),
    component: str = typer.Option("vision", help="Component: full, vision, action_head"),
    output_dir: Path = typer.Option(Path("./exported"), help="Output directory"),
) -> None:
    """Export a VLA model (or component) to optimized format."""
    from vla_edge.optimize.onnx_export import export_to_onnx
    from vla_edge.registry import get_model

    if format == "gguf":
        from vla_edge.optimize.quantize import quantize_to_gguf, recommend_quantization

        console.print(f"[bold]Quantizing[/bold] {model} to GGUF...")
        try:
            model_cls = get_model(model)
            info = model_cls.model_info()
            rec = recommend_quantization(info.param_count)
            console.print(f"  Recommended quantization for {info.param_count / 1e9:.1f}B params: {rec}")
        except (KeyError, AttributeError):
            pass

        result = quantize_to_gguf(model, output_dir, quantization=component or "q4_k_m")
        if result.success:
            console.print(f"[green]Quantized to {result.quantized_path}[/green]")
            console.print(
                f"  {result.original_size_mb}MB -> {result.quantized_size_mb}MB ({result.compression_ratio}x)"
            )
        else:
            console.print(f"[red]{result.error}[/red]")
        return

    console.print(f"[bold]Exporting[/bold] {model} ({component}) to {format}...")

    try:
        model_cls = get_model(model)
        loaded = model_cls(device="cpu")
        loaded._ensure_loaded()
    except (KeyError, TypeError) as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    result = export_to_onnx(loaded, output_dir, component=component)

    if result.success:
        console.print(f"[green]Exported to {result.output_path}[/green]")
        console.print(f"  Original: {result.original_size_mb:.1f} MB")
        console.print(f"  Exported: {result.exported_size_mb:.1f} MB")
    else:
        console.print(f"[red]Export failed: {result.error}[/red]")

    if hasattr(loaded, "cleanup"):
        loaded.cleanup()


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
