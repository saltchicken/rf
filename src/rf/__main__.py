import uvicorn
import os
import subprocess
import argparse
from pathlib import Path

frontend_dir = f"{Path(__file__).parent}/frontend"
current_dir = Path(__file__).parent
print(f"Current directory: {current_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run or build the frontend.")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        help="Command to run (e.g., 'build' or 'run')",
    )

    args = parser.parse_args()

    if args.command == "build":
        subprocess.run(["bun", "install"], cwd=frontend_dir)
        subprocess.run(["bun", "run", "build"], cwd=frontend_dir)

        # For example: os.system("npm run build")
    else:
        # os.chdir(frontend_dir)
        uvicorn.run("rf.frontend.main:app", host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
