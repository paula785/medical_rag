"""
MedRAG Application Launcher
----------------------------
Clean, production-grade orchestrator that ensures background services (Ollama daemon)
are active before launching the Streamlit web application.
"""

import os
import sys
import time
import socket
import subprocess

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Non-blocking TCP socket check to verify if Ollama server port is bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False

def ensure_ollama_service() -> bool:
    """Verifies Ollama service health, launching 'ollama serve' in background if offline."""
    print(f"🔍 Checking Ollama daemon at http://{OLLAMA_HOST}:{OLLAMA_PORT}...")
    if is_port_open(OLLAMA_HOST, OLLAMA_PORT):
        print("✅ Ollama service is active and listening.")
        return True

    print("🚀 Ollama service not detected. Starting 'ollama serve' in background...")
    try:
        if sys.platform == "win32":
            creationflags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=creationflags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        # Poll socket up to 8 seconds for initialization
        for attempt in range(1, 9):
            time.sleep(1)
            if is_port_open(OLLAMA_HOST, OLLAMA_PORT):
                print(f"✅ Ollama service initialized in {attempt}s!")
                return True

        print("⚠️ Ollama launch initiated, but service did not respond within 8s. App will launch with fallback UI.")
        return False
    except Exception as err:
        print(f"❌ Failed to auto-start Ollama: {err}")
        return False

def main():
    ensure_ollama_service()
    print("🩺 Launching MedRAG Web Interface...")
    
    # Launch Streamlit app using current Python executable
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 MedRAG Application shut down cleanly.")

if __name__ == "__main__":
    main()
