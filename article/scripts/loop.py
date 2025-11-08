#!/usr/bin/env python3
import argparse
import difflib
import json
import subprocess
import sys
import pathlib
import time
from datetime import datetime
from typing import Optional

try:
    import yaml  # pip install pyyaml
except ImportError:
    sys.exit("Please `pip install pyyaml` to use article/scripts/loop.py")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "article/config/loop.yaml").read_text())
PROMPT_VALUES = {
    "PROJECT_NAME": CFG["project"]["name"],
    "RESULTS_DIR": CFG["paths"]["results_dir"],
    "MANUSCRIPT_MAIN": CFG["paths"]["manuscript_main"],
    "MANUSCRIPT_SUP": CFG["paths"]["manuscript_sup"],
    "BIB_PATH": CFG["paths"]["references_bib"],
    "SPG_PATH": CFG["paths"]["spg"],
    "OUTLINE_PATH": CFG["paths"]["outline"],
    "DOCS_DIR": CFG["paths"]["docs_dir"],
    "OUTLINE_STRATEGY": CFG["planner"]["outline_strategy"],
    "IMAGES_DIR": CFG["paths"]["images_dir"],
    "REVIEW_DIR": "article/artifacts",
    "AUTHOR_GUIDELINES": "",
}
GUIDELINES_DIR = CFG["paths"].get("guidelines_dir")
CODEX_CONFIG_PATH = ROOT / ".tooling" / "codex" / "config.toml"


def load_codex_overrides():
    if not CODEX_CONFIG_PATH.exists():
        return []
    overrides = []
    current_section = None
    for raw_line in CODEX_CONFIG_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        full_key = f"{current_section}.{key.strip()}" if current_section else key.strip()
        overrides.extend(["-c", f"{full_key}={value.strip()}"])
    return overrides


CODEX_OVERRIDES = load_codex_overrides()
STEP_INDEX = {"codex": 1, "claude": 2, "gemini": 3}


def run(cmd, cwd=ROOT, interactive=False):
    print(">>", " ".join(cmd), flush=True)
    if interactive:
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode != 0:
            sys.exit(r.returncode)
        return ""

    r = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    return r.stdout


def render_template(path):
    text = pathlib.Path(path).read_text()
    for key, value in PROMPT_VALUES.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def load_author_guidelines() -> list[str]:
    if not GUIDELINES_DIR:
        return []
    base = ROOT / GUIDELINES_DIR
    if not base.exists():
        return []
    parts: list[str] = []
    for candidate in sorted(base.rglob("*.md")):
        try:
            text = candidate.read_text().strip()
            if text:
                parts.append(text)
        except OSError:
            continue
    return parts


def latest_review_blob():
    artifacts_dir = ROOT / "article" / "artifacts"
    if not artifacts_dir.exists():
        return None
    candidates = sorted(artifacts_dir.glob("review_*.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        content = latest.read_text().strip()
    except OSError:
        return None
    return latest.name, content


def extract_review_from_stream(stream_text: str) -> Optional[str]:
    """Parse Gemini stream-json output and return the embedded review JSON."""
    if not stream_text:
        return None
    assembled = []
    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # attempt to parse top-level object
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "message" and obj.get("role") == "assistant":
            content = obj.get("content", "")
            if content:
                assembled.append(content)
    if not assembled:
        return None
    payload = "".join(assembled)
    if "```json" in payload:
        start = payload.index("```json") + len("```json")
        end = payload.find("```", start)
        if end == -1:
            body = payload[start:]
        else:
            body = payload[start:end]
        return body.strip()
    # fall back: payload itself may already be JSON
    payload = payload.strip()
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        return payload


def codex(prompt_path, mode="headless"):
    prompt = render_template(prompt_path)

    if mode == "interactive":
        print(f"[codex] Prompt from {prompt_path}:\n{prompt}\n")
        cmd = ["codex", *CODEX_OVERRIDES, "exec", "--cd", str(ROOT), "--", prompt]
        return run(cmd, interactive=True)

    cmd = [
        "codex",
        *CODEX_OVERRIDES,
        "exec",
        "--cd",
        str(ROOT),
        "--json",
        "--",
        prompt,
    ]
    return run(cmd)


def claude(prompt_path, mode="headless"):
    prompt = render_template(prompt_path)
    review_info = latest_review_blob()
    if review_info:
        filename, blob = review_info
        prompt = (
            f"{prompt}\n\n"
            f"---\nLatest Gemini review (`{filename}`):\n```json\n{blob}\n```\n"
            "Address each point (or log TODOs) before handing off.\n"
        )

    base_args = [
        "--allow-dangerously-skip-permissions",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "acceptEdits",
    ]

    if mode == "interactive":
        print(f"[claude] Prompt from {prompt_path}:\n{prompt}\n")
        cmd = [
            "claude",
            "--verbose",
            *base_args,
            "-p",
            "--output-format",
            "stream-json",
            prompt,
        ]
        print(
            "+ streaming Claude output (ctrl+C to abort; transcript logged above)\n",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        buffer: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                buffer.append(line)
        finally:
            proc.wait()
        if proc.returncode != 0:
            sys.exit(proc.returncode)
        return "".join(buffer)

    return run(
        ["claude", "--verbose", *base_args, "-p", "--output-format", "json", prompt],
    )


def gemini(prompt_path, mode="headless"):
    prompt = render_template(prompt_path)

    if mode == "interactive":
        print(f"[gemini] Prompt from {prompt_path}:\n{prompt}\n")
        cmd = ["gemini", "--output-format", "stream-json", prompt]
        print(
            "+ streaming Gemini output (ctrl+C to abort; review will be saved when complete)\n",
            flush=True,
        )
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT / "article",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        buffer = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                buffer.append(line)
        finally:
            proc.wait()
        stream_text = "".join(buffer)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
        extracted = extract_review_from_stream(stream_text)
        return extracted if extracted else stream_text

    cmd = ["gemini", "--output-format", "json", prompt]
    return run(cmd, cwd=ROOT / "article")


def build_pdf():
    try:
        run(["latexmk", "-pdf", "-quiet", CFG["paths"]["manuscript_main"]], cwd=ROOT)
    except FileNotFoundError:
        print("[loop] latexmk not available; skipping PDF build.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Codex → Claude → Gemini loop orchestrator")
    parser.add_argument(
        "--mode",
        choices=["headless", "interactive"],
        default="headless",
        help="Run automation in headless mode (default) or launch interactive CLIs.",
    )
    parser.add_argument(
        "--start-from",
        choices=["codex", "claude", "gemini"],
        default="codex",
        help="Skip agents before the selected stage.",
    )
    parser.add_argument(
        "--build-pdf",
        action="store_true",
        help="Run latexmk to compile the manuscript after Claude edits.",
    )
    parser.add_argument(
        "--guideline",
        action="append",
        default=[],
        help="Inline author guideline (can be provided multiple times).",
    )
    parser.add_argument(
        "--guideline-file",
        action="append",
        default=[],
        help="Path to a text/markdown file containing author guidelines (repeatable).",
    )
    args = parser.parse_args()
    mode = args.mode
    initial_start_index = STEP_INDEX[args.start_from]

    iters = int(CFG["loop"]["max_iters"])
    gates = CFG.get("readability_gates", {})
    clarity_min = float(gates.get("clarity_min", 0))
    storytelling_min = float(gates.get("storytelling_min", 0))
    readability_min = float(gates.get("readability_min", 0))
    max_conf = int(gates.get("max_confusions", sys.maxsize))
    max_style = int(gates.get("max_style_findings", sys.maxsize))

    guideline_chunks: list[str] = []
    for inline in args.guideline:
        text = inline.strip()
        if text:
            guideline_chunks.append(text)
    for file_path in args.guideline_file:
        try:
            text = (ROOT / file_path).read_text().strip() if not pathlib.Path(file_path).is_absolute() else pathlib.Path(file_path).read_text().strip()
            if text:
                guideline_chunks.append(text)
        except OSError as exc:
            print(f"[loop] warning: failed to read guideline file '{file_path}': {exc}", file=sys.stderr)
    guideline_chunks.extend(load_author_guidelines())
    combined_guidelines = "\n\n".join(guideline_chunks)
    if not combined_guidelines.strip():
        combined_guidelines = "(no explicit author guidelines provided)"
    PROMPT_VALUES["AUTHOR_GUIDELINES"] = combined_guidelines

    manuscript_relpaths = [
        pathlib.Path(PROMPT_VALUES["MANUSCRIPT_MAIN"]),
        pathlib.Path(PROMPT_VALUES["MANUSCRIPT_SUP"]),
        pathlib.Path(PROMPT_VALUES["BIB_PATH"]),
    ]
    manuscript_paths = [ROOT / rel for rel in manuscript_relpaths]
    original_manuscript: dict[pathlib.Path, Optional[str]] = {}
    for abs_path in manuscript_paths:
        try:
            original_manuscript[abs_path] = abs_path.read_text()
        except OSError:
            original_manuscript[abs_path] = None

    timings = {
        "codex_graph": 0.0,
        "codex_outline": 0.0,
        "claude": 0.0,
        "build_pdf": 0.0,
        "gemini": 0.0,
    }
    overall_start = time.time()

    for i in range(1, iters + 1):
        print(f"\n==== LOOP {i} ({mode}) ====\n")
        loop_start_index = initial_start_index if i == 1 else STEP_INDEX["codex"]

        if loop_start_index <= STEP_INDEX["codex"]:
            t0 = time.time()
            codex("article/scripts/prompts/codex_graph.txt", mode=mode)
            timings["codex_graph"] += time.time() - t0
            t0 = time.time()
            codex("article/scripts/prompts/codex_linearize.txt", mode=mode)
            timings["codex_outline"] += time.time() - t0
        else:
            print("[loop] Skipping Codex stages (start-from>=claude).")

        if loop_start_index <= STEP_INDEX["claude"]:
            t0 = time.time()
            claude("article/scripts/prompts/claude_draft.txt", mode=mode)
            timings["claude"] += time.time() - t0
        else:
            print("[loop] Skipping Claude stage (start-from=gemini).")

        if args.build_pdf:
            t0 = time.time()
            build_pdf()
            timings["build_pdf"] += time.time() - t0

        t0 = time.time()
        review_json = gemini("article/scripts/prompts/gemini_blind_review.txt", mode=mode)
        timings["gemini"] += time.time() - t0

        try:
            review = json.loads(review_json)
        except Exception:
            review = json.loads(review_json.strip().splitlines()[-1])

        artifacts_dir = ROOT / "article" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        review_path = artifacts_dir / f"review_loop{i}_{timestamp}.json"
        review_path.write_text(json.dumps(review, indent=2))

        confs = len(review.get("confusions", []))
        clarity = review.get("clarity_score", 0)
        storytelling = review.get("storytelling_score", 0)
        readability = review.get("readability_score", 0)
        style_findings = review.get("style_findings", [])
        style_count = len(style_findings) if isinstance(style_findings, list) else 0
        if (
            CFG["loop"]["stop_on_success"]
            and clarity >= clarity_min
            and storytelling >= storytelling_min
            and readability >= readability_min
            and confs <= max_conf
            and style_count <= max_style
        ):
            print(
                "[loop] stop: "
                f"clarity={clarity} storytelling={storytelling} "
                f"readability={readability} confusions={confs} "
                f"style_findings={style_count}"
            )
            break
    else:
        print("[loop] finished without meeting gates")

    print("\n==== Manuscript Diff Summary ====")
    for rel_path, abs_path in zip(manuscript_relpaths, manuscript_paths):
        before = original_manuscript.get(abs_path)
        try:
            after = abs_path.read_text()
        except OSError:
            after = None

        display_name = str(rel_path)
        if before is None and after is None:
            print(f"- {display_name}: unavailable")
            continue
        if before is None:
            added_lines = len(after.splitlines()) if after is not None else 0
            print(f"- {display_name}: created ({added_lines} lines)")
            continue
        if after is None:
            print(f"- {display_name}: deleted")
            continue
        if before == after:
            print(f"- {display_name}: no textual changes")
            continue

        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"before/{display_name}",
                tofile=f"after/{display_name}",
                lineterm="",
            )
        )
        max_lines = 120
        truncated = len(diff_lines) > max_lines
        snippet = diff_lines[:max_lines]
        print(f"- {display_name} diff:")
        for line in snippet:
            print(f"  {line}")
        if truncated:
            print(f"  ... ({len(diff_lines) - max_lines} more diff lines)")

    total_elapsed = time.time() - overall_start
    print("\n==== Timing Summary ====")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    for key, secs in timings.items():
        print(f"- {key}: {secs:.1f}s")


if __name__ == "__main__":
    main()
