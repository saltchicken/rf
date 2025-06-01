import uvicorn
import os


def main():
    os.chdir("frontend")
    uvicorn.run("frontend.main:app", host="127.0.0.1", port=5000)


if __name__ == "__main__":
    main()
