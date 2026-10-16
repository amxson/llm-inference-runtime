"""Derive a standalone SVG from final measurement JSON; no hand-copied timings."""
import json
from pathlib import Path
import statistics


def mean_positions(rows):
    return [statistics.mean(column) for column in zip(*rows)]


def render(data):
    series = [
        ("Uncached local selection", mean_positions([r["inter_token_latencies_ms"] for r in data["paired"]["uncached"]["runs"]]), "#CFCBC2"),
        ("Cached local selection", mean_positions([r["inter_token_latencies_ms"] for r in data["paired"]["cached"]["runs"]]), "#EFA23A"),
    ]
    network = [("Uncached HTTP arrival (separate scope)", mean_positions([r["token_gaps_ms"] for r in data["network_runs"]]), "#CFCBC2")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="720" viewBox="0 0 1000 720" role="img" aria-label="Measured token gap against token position">',
           '<rect width="1000" height="720" fill="#0E0F10"/>',
           '<g fill="#E8E6E1" font-family="sans-serif" font-size="14">',
           '<text x="40" y="32" font-size="20">Token-level timing / final CPU rerun</text>']
    for top, curves in [(80, series), (390, network)]:
        maximum = max(v for _, values, _ in curves for v in values)
        for fraction in (0, .5, 1):
            y = top + 210 - fraction * 180
            svg.append(f'<path d="M80 {y}H940" stroke="#25272B"/><text x="16" y="{y+4}">{maximum*fraction:.1f}</text>')
        for j, (name, values, color) in enumerate(curves):
            svg.append(f'<text x="{80+j*380}" y="{top}" fill="{color}">{name}</text>')
            points = " ".join(f"{80+i/(len(values)-1)*860:.2f},{top+210-v/maximum*180:.2f}" for i, v in enumerate(values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        svg.extend([f'<text x="80" y="{top+236}">2</text>', f'<text x="916" y="{top+236}">128</text>',
                    f'<text x="350" y="{top+258}">Output token position (TTFT excluded)</text>', f'<text x="20" y="{top+25}">ms</text>'])
    svg.extend(['<text x="40" y="692">Mean per position / 5 runs / 2 excluded warmups. HTTP and CLI timings are not interchangeable.</text>', '</g></svg>'])
    return "\n".join(svg) + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "benchmarks/results/final_cpu.json").read_text(encoding="utf8"))
    (root / "docs/images/final-token-gaps.svg").write_text(render(data), encoding="utf8")
