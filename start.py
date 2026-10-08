import sys
import webbrowser
import threading
import time
from pathlib import Path
import uvicorn

def open_browser():
    time.sleep(1.5)
    print("\n[OmniRAG] Opening web application in browser: http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))

    print("=" * 65)
    print("      Starting OmniRAG: Universal Document Intelligence")
    print("=" * 65)
    print(f"Project Location : {current_dir}")
    print("Server URL       : http://127.0.0.1:8000")
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[str(current_dir / "backend")])
