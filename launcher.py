"""
launcher.py — Single-Entry-Point Launcher for Secure AI Agent Chatbot
----------------------------------------------------------------------
Automatically detects running services (FastAPI on 8000, Streamlit on 8501),
launches missing components using the project's virtual environment, opens
the browser, and cleanly manages process lifecycles.
"""

import os
import sys
import time
import socket
import urllib.request
import subprocess
import webbrowser
import signal
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("launcher")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "Scripts", "python.exe")
PYTHON_EXE = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501

SPAWNED_PROCESSES = []


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open and accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def is_backend_healthy() -> bool:
    """Check if the FastAPI backend /health endpoint returns 200 OK."""
    try:
        url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "SecureAgentLauncher/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def cleanup_spawned_processes(*args):
    """Cleanly terminate only the child processes spawned by this launcher."""
    logger.info("🛑 Shutting down spawned launcher processes...")
    for proc in SPAWNED_PROCESSES:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
                logger.info(f"✅ Terminated process PID {proc.pid}")
            except Exception:
                try:
                    proc.kill()
                    logger.info(f"⚡ Killed process PID {proc.pid}")
                except Exception as e:
                    logger.warning(f"Failed to kill PID {proc.pid}: {e}")
    sys.exit(0)


def launch():
    signal.signal(signal.SIGINT, cleanup_spawned_processes)
    signal.signal(signal.SIGTERM, cleanup_spawned_processes)

    logger.info("🚀 Starting Secure AI Agent Chatbot Launcher...")
    logger.info(f"📁 Project Directory: {PROJECT_DIR}")
    logger.info(f"🐍 Python Executable: {PYTHON_EXE}")

    # 1. Check & Start FastAPI Backend
    backend_was_running = is_backend_healthy()
    if backend_was_running:
        logger.info(f"✅ FastAPI backend is already running on http://{BACKEND_HOST}:{BACKEND_PORT}")
    else:
        logger.info(f"⚙️ Launching FastAPI backend on http://{BACKEND_HOST}:{BACKEND_PORT}...")
        backend_cmd = [
            PYTHON_EXE, "-m", "uvicorn", "main:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT)
        ]
        backend_proc = subprocess.Popen(backend_cmd, cwd=PROJECT_DIR)
        SPAWNED_PROCESSES.append(backend_proc)

        # Wait for backend readiness
        max_attempts = 30
        logger.info("⏳ Waiting for backend to become healthy...")
        for attempt in range(max_attempts):
            time.sleep(1)
            if is_backend_healthy():
                logger.info("✅ FastAPI backend is online and healthy!")
                break
        else:
            logger.error("❌ Backend failed to start within 30 seconds.")
            cleanup_spawned_processes()

    # 2. Check & Start Streamlit Frontend
    frontend_was_running = is_port_open(BACKEND_HOST, FRONTEND_PORT)
    if frontend_was_running:
        logger.info(f"✅ Streamlit UI is already running on http://{BACKEND_HOST}:{FRONTEND_PORT}")
    else:
        logger.info(f"🖥️ Launching Streamlit UI on http://{BACKEND_HOST}:{FRONTEND_PORT}...")
        frontend_cmd = [
            PYTHON_EXE, "-m", "streamlit", "run", "app_ui.py",
            "--server.port", str(FRONTEND_PORT),
            "--server.address", BACKEND_HOST,
            "--server.headless", "true"
        ]
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=PROJECT_DIR)
        SPAWNED_PROCESSES.append(frontend_proc)

        # Wait for Streamlit port to open
        logger.info("⏳ Waiting for Streamlit UI readiness...")
        for _ in range(15):
            time.sleep(1)
            if is_port_open(BACKEND_HOST, FRONTEND_PORT):
                logger.info("✅ Streamlit UI is ready!")
                break

    # 3. Open Browser
    url = f"http://{BACKEND_HOST}:{FRONTEND_PORT}"
    logger.info(f"🌐 Opening chatbot interface in browser: {url}")
    webbrowser.open(url)

    logger.info("✨ Secure AI Agent Chatbot is running!")
    logger.info("Press Ctrl+C in this terminal window to stop the application.")

    # Keep launcher process alive if it spawned any child processes
    if SPAWNED_PROCESSES:
        try:
            while True:
                time.sleep(2)
                # Check if spawned processes are still alive
                for proc in SPAWNED_PROCESSES:
                    if proc.poll() is not None:
                        logger.warning(f"⚠️ Child process PID {proc.pid} exited with code {proc.returncode}")
        except KeyboardInterrupt:
            cleanup_spawned_processes()


if __name__ == "__main__":
    launch()
