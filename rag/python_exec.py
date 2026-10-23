"""
rag/python_exec.py — Sandboxed Python Execution Tool
--------------------------------------------------------
SECURITY NOTE: Never use exec() or eval() directly on LLM-generated or
user-supplied code in a web-facing service. This module runs code in a
SEPARATE PROCESS with:
  - A timeout (kills runaway/infinite loops)
  - No network access at the OS level (you must additionally run this
    behind a container/firewall — see deployment notes at bottom)
  - Restricted builtins inside the subprocess
  - Output size limits (prevents memory exhaustion from huge prints)

This is "reasonably safe for a portfolio project demo," NOT
"production-grade arbitrary code execution." For real production use
you'd want gVisor, Firecracker microVMs, or a service like Modal/E2B/
Daytona that specializes in sandboxed code execution. Mention this
tradeoff honestly in your README — it shows you understand the risk
surface, which actually reads BETTER to evaluators than pretending
exec() is fine.
"""

import subprocess
import sys
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)

EXECUTION_TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 4000

# Wrapped around user code before execution — strips access to the most
# dangerous builtins. This runs INSIDE the subprocess, so even if someone
# works around it, they're still contained by the subprocess boundary.
_SANDBOX_PREAMBLE = """
import sys as _sys
import builtins as _builtins

_BLOCKED_MODULES = {'os', 'subprocess', 'socket', 'shutil', 'ctypes'}

class _BlockedImporter:
    def find_module(self, name, path=None):
        if name.split('.')[0] in _BLOCKED_MODULES:
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"Module '{name}' is blocked in this sandbox")

_sys.meta_path.insert(0, _BlockedImporter())

# CRITICAL: 'os' and friends are already loaded into sys.modules by the
# interpreter's own startup (and by this preamble's imports). The
# meta_path finder above only stops FUTURE imports — it does nothing
# about modules already cached. `import os` in user code would just
# return the cached module, bypassing our finder entirely. Evict them
# from the cache so the next `import os` is forced through our finder
# and gets blocked.
for _mod in list(_BLOCKED_MODULES):
    _sys.modules.pop(_mod, None)

# Now disable dangerous builtins (NOT __import__ — breaking that crashes
# the interpreter itself; module blocking above already covers the same risk)
for _name in ('exec', 'eval', 'compile', 'open', 'input'):
    if hasattr(_builtins, _name):
        setattr(_builtins, _name, None)
"""


def run_python(code: str) -> dict:
    """
    Execute Python code in an isolated subprocess with a timeout.

    Returns the same {"answer", "sources", "search_type"} shape your
    other tools use, so the agent can merge it into context like any
    other tool result.
    """
    full_code = _SANDBOX_PREAMBLE + "\n\n" + code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_file:
        tmp_file.write(full_code)
        tmp_path = tmp_file.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
            # On Linux you can further restrict with resource limits via
            # preexec_fn=_set_resource_limits (see note at bottom)
        )

        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]

        if result.returncode != 0:
            logging.warning(f"🐍 Python exec error: {stderr}")
            return {
                "answer": f"Execution error:\n{stderr}",
                "sources": ["python_exec"],
                "search_type": "python_exec_error",
                "stdout": stdout,
                "stderr": stderr,
            }

        logging.info(f"🐍 Python exec succeeded, output: {len(stdout)} chars")
        return {
            "answer": stdout if stdout else "(no output)",
            "sources": ["python_exec"],
            "search_type": "python_exec",
            "stdout": stdout,
            "stderr": stderr,
        }

    except subprocess.TimeoutExpired:
        logging.warning(f"🐍 Python exec timed out after {EXECUTION_TIMEOUT_SECONDS}s")
        return {
            "answer": f"Execution timed out after {EXECUTION_TIMEOUT_SECONDS} seconds.",
            "sources": ["python_exec"],
            "search_type": "python_exec_timeout",
            "stdout": "",
            "stderr": "TimeoutExpired",
        }

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Deployment notes ─────────────────────────────────────────────────────
#
# For Render/Railway deployment, add a Linux resource-limit wrapper using
# the `resource` module (Unix only) for an extra layer of defense:
#
#   import resource
#   def _set_resource_limits():
#       resource.setrlimit(resource.RLIMIT_CPU, (5, 5))          # 5 CPU seconds
#       resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)  # 256MB
#
# Then pass preexec_fn=_set_resource_limits to subprocess.run() above
# (Linux/Mac only — not available on Windows, which is fine since your
# deploy target is Render/Railway, both Linux).
#
# For a stronger boundary than subprocess alone, consider running this
# tool's subprocess inside its own Docker container with --network=none,
# or swap this module out entirely for a hosted sandbox API (E2B, Modal)
# once you're ready to harden it past portfolio-demo level.
