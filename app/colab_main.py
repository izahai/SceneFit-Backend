import os
import time
import subprocess
from pyngrok import ngrok

# -------------------------------------------------
# Config
# -------------------------------------------------
PORT = 8000
RUN_CMD = ["bash", "run.sh"]  # or ["python", "-m", "uvicorn", "app.main:app"]

# Optional: use env var for safety
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN")

# -------------------------------------------------
# Start FastAPI server
# -------------------------------------------------
print("Starting FastAPI server...")
server_proc = subprocess.Popen(
    RUN_CMD,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

# Wait for server to boot
time.sleep(3)

# -------------------------------------------------
# Start ngrok tunnel
# -------------------------------------------------
if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)

public_url = ngrok.connect(PORT)
print(f"Public API URL: {public_url}")
print(f"Swagger UI:    {public_url}/docs\n")

# -------------------------------------------------
# Keep everything alive
# -------------------------------------------------
try:
    print("Server + ngrok running. Press CTRL+C to stop.\n")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
ngrok.kill()
server_proc.terminate()
