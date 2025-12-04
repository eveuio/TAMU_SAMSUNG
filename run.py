
#TODO: import streamlit (streamlitApp.py) and fast API (fast.py) scripts from  /TAMU_SAMSUNG/frontEnd. 
# Original function calls when running from /TAMU_SAMSUNG/frontEnd: 
#       fast.py:           PYTHONPATH=.. uvicorn fast:app --reload --host 127.0.0.1 --port 8000
#       strealitApp.py:    streamlit run streamlitApp.py

#TODO: Terminal call FastAPI first as a background process, and the streamlit

#TODO: If program terminated, terminate FastAPI


import os
import subprocess
import sys
import time
import signal
import importlib.util
import webbrowser

# Paths
CWD = os.getcwd()
FRONTEND_DIR = os.path.join(CWD, "frontEnd")
FAST_MODULE = "fast:app"
STREAMLIT_FILE = os.path.join(FRONTEND_DIR, "streamlitApp.py")

def ensure_module_available(mod_name: str, friendly: str):
    """Check if a module is installed, else exit with instructions."""
    if importlib.util.find_spec(mod_name) is None:
        print(
            f"[Launcher] ERROR: '{friendly}' is not installed.\n"
            f"Install it with:\n    {sys.executable} -m pip install {friendly}\n"
        )
        sys.exit(1)

def start_fastapi():
    """Start FastAPI using uvicorn with PYTHONPATH set to parent directory."""
    ensure_module_available("uvicorn", "uvicorn")

    env = os.environ.copy()
    parent_dir = os.path.dirname(FRONTEND_DIR)
    env["PYTHONPATH"] = f"{parent_dir}" if "PYTHONPATH" not in env else f"{parent_dir}{os.pathsep}{env['PYTHONPATH']}"

    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn",
        FAST_MODULE,
        "--reload",
        "--host", "127.0.0.1",
        "--port", "8000",
    ]

    print(f"[Launcher] Starting FastAPI in {FRONTEND_DIR} with PYTHONPATH={env['PYTHONPATH']}")
    fastapi_proc = subprocess.Popen(
        uvicorn_cmd,
        cwd=FRONTEND_DIR,
        env=env
    )
    return fastapi_proc

def start_streamlit():
    """Start Streamlit and auto-open browser."""
    ensure_module_available("streamlit", "streamlit")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", STREAMLIT_FILE,
        "--server.headless", "true",  
        "--browser.serverAddress", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    print(f"[Launcher] Starting Streamlit: {' '.join(streamlit_cmd)}")
    streamlit_proc = subprocess.Popen(
        streamlit_cmd,
        cwd=FRONTEND_DIR
    )
    # Wait for Streamlit to start, then open browser
    time.sleep(5)
    webbrowser.open("http://localhost:8501")
    return streamlit_proc

def main():
    fastapi_proc = None
    streamlit_proc = None

    try:
        if not os.path.isdir(FRONTEND_DIR):
            print(f"[Launcher] ERROR: frontEnd directory not found at {FRONTEND_DIR}")
            sys.exit(1)
        if not os.path.isfile(STREAMLIT_FILE):
            print(f"[Launcher] ERROR: streamlitApp.py not found at {STREAMLIT_FILE}")
            sys.exit(1)

        fastapi_proc = start_fastapi()
        time.sleep(2)  # give FastAPI time to bind

        streamlit_proc = start_streamlit()

        # Wait until Streamlit exits
        streamlit_proc.wait()

    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted (Ctrl+C). Shutting down...")
    finally:
        # Cleanup Streamlit
        if streamlit_proc and streamlit_proc.poll() is None:
            try:
                print("[Launcher] Terminating Streamlit...")
                streamlit_proc.terminate()
                streamlit_proc.wait(timeout=5)
            except Exception:
                try:
                    streamlit_proc.kill()
                except Exception:
                    pass

        # Cleanup FastAPI
        if fastapi_proc and fastapi_proc.poll() is None:
            try:
                print("[Launcher] Terminating FastAPI...")
                if os.name == "posix":
                    fastapi_proc.send_signal(signal.SIGINT)
                    time.sleep(1)
                fastapi_proc.terminate()
                fastapi_proc.wait(timeout=10)
            except Exception:
                try:
                    fastapi_proc.kill()
                except Exception:
                    pass

        print("[Launcher] All processes stopped.")

if __name__ == "__main__":
    main()
