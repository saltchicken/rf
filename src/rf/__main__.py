import uvicorn
import os

from pathlib import Path

frontend_dir = f"{Path(__file__).parent}/frontend"

current_dir = Path(__file__).parent
print(f"Current directory: {current_dir}")


def main():
    os.chdir(frontend_dir)
    uvicorn.run("rf.frontend.main:app", host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
