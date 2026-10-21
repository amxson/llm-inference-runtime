"""Plan first, then deliberately install two NEW venvs; never edit base Colab."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

from common import ROOT, child_environment, save_json, source_provenance, utc_now


def install_commands(env_root):
    index = ["--index-url", "https://pypi.org/simple", "--extra-index-url", "https://download.pytorch.org/whl/cu129",
             "--only-binary=:all:"]
    return {
        "runtime": [str(env_root / "runtime/bin/python"), "-m", "pip", "install", *index,
                    "torch==2.10.0+cu129", "transformers==5.17.0", "-e", ".[server,test]"],
        "vllm": [str(env_root / "vllm/bin/python"), "-m", "pip", "install", *index,
                 "torch==2.10.0+cu129", "torchaudio==2.10.0+cu129", "torchvision==0.25.0+cu129",
                 "transformers==4.57.6", "vllm==0.17.1"]}


def logged(command, path):
    print("Running:", " ".join(command), flush=True)
    with Path(path).open("x", encoding="utf8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    if code:
        print(Path(path).read_text(encoding="utf8")[-12000:], flush=True)
        raise RuntimeError(f"Command exited {code}. Stop here and return {path}; do not patch dependencies.")


def bootstrap_venv(directory, output):
    """Do not depend on Debian/Colab's optional ensurepip installation.

    pip >=22.3 can manage a pip-less target venv via --python. The host pip
    runs the command, but installation goes ONLY into the explicit new venv.
    No apt install, base pip upgrade or system-site-packages is needed.
    """
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(directory)
    logged([sys.executable, "-m", "pip", "--python", str(directory), "install",
            "--index-url", "https://pypi.org/simple", "--only-binary=:all:", "pip==26.2.2"],
           output / f"bootstrap-{directory.name}.txt")


def resolved_constraints(report):
    """Lock the binary package versions selected by pip's reviewed dry run."""
    return "".join(sorted(
        f"{entry['metadata']['name']}=={entry['metadata']['version']}\n"
        for entry in report["install"]
        if entry["metadata"]["name"] != "llm-inference-runtime"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Execute the already-recorded compatible resolution plans")
    args = parser.parse_args()
    if sys.platform != "linux" or not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error("Use Colab Linux with Python 3.11–3.13; do not alter the Windows environment")
    provenance = source_provenance()
    plan_path = args.output / "installation-plan.json"
    commands = install_commands(args.env_root)
    if not args.install:
        if args.env_root.exists() or args.output.exists():
            parser.error("Choose fresh environment and evidence directories; existing work is never deleted")
        if shutil.disk_usage("/content").free < 20 * 1024**3:
            parser.error("Reserve at least 20 GiB free disk for this two-environment installation plan; this is an estimate, not a measured minimum")
        args.output.mkdir(parents=True)
        args.env_root.mkdir(parents=True)
        constraints_hashes = {}
        for name in commands:
            bootstrap_venv(args.env_root / name, args.output)
            logged(commands[name] + ["--dry-run", "--report", str(args.output / f"resolve-{name}.json")],
                   args.output / f"resolve-{name}.txt")
            report = json.loads((args.output / f"resolve-{name}.json").read_text(encoding="utf8"))
            constraints = args.output / f"constraints-{name}.txt"
            with constraints.open("x", encoding="utf8") as target:
                target.write(resolved_constraints(report))
            constraints_hashes[name] = hashlib.sha256(constraints.read_bytes()).hexdigest()
        save_json(plan_path, {"recorded_at_utc": utc_now(), **provenance, "commands": commands,
                              "constraints_sha256": constraints_hashes,
                              "base_python": platform.python_version(),
                              "base_environment_untouched": True,
                              "note": "pip resolution succeeds before install; this does NOT prove CUDA/vLLM runtime compatibility"})
        print(f"Resolution plans ready in {args.output}. Review both text/JSON reports, then run --install.")
        return
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if plan["commands"] != commands or plan["source_commit"] != provenance["source_commit"]:
        parser.error("Plan does not match this checkout/environment; create a new plan")
    for name, command in commands.items():
        constraints = args.output / f"constraints-{name}.txt"
        if hashlib.sha256(constraints.read_bytes()).hexdigest() != plan["constraints_sha256"][name]:
            parser.error("Resolution constraints changed; create a new plan instead")
        logged(command + ["--constraint", str(constraints), "--report", str(args.output / f"installed-{name}.json")],
               args.output / f"install-{name}.txt")
        python = command[0]
        logged([python, "-m", "pip", "check"], args.output / f"pip-check-{name}.txt")
        logged([python, "-m", "pip", "freeze", "--all"], args.output / f"freeze-{name}.txt")
        logged([python, "scripts/phase3/preflight.py", "--output", str(args.output / f"environment-{name}.json")],
               args.output / f"preflight-{name}.txt")
    print("Both isolated installations, pip checks and real FP16 CUDA preflights passed.", flush=True)


if __name__ == "__main__":
    main()
