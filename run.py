"""
MedRAG Unified Multi-Service Launcher
-------------------------------------
Launches FastAPI backend on port 8000 and Streamlit frontend on port 8501 cleanly.
"""

import sys
import os
import time
import subprocess

def main():
    print("🚀 Starting MedRAG Decoupled FastAPI + Streamlit System...")
    
    # 1. Launch FastAPI Uvicorn Server in Background (No CMD window)
    server_cmd = [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"]
    print("⚡ Starting FastAPI Microservice on http://127.0.0.1:8000...")
    
    if sys.platform == "win32":
        creationflags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
        backend_proc = subprocess.Popen(server_cmd, creationflags=creationflags)
    else:
        backend_proc = subprocess.Popen(server_cmd, start_new_session=True)

    time.sleep(2.5)

    # 2. Launch Streamlit UI
    print("🩺 Starting Streamlit Web Interface on http://localhost:8501...")
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\n👋 Shutting down MedRAG services...")
    finally:
        try:
            backend_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
