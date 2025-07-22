# ReaderX
PDF parsing tool

## Docker

Run this project in Docker:

1. **Build the image:**
   ```sh
   docker build -t pdf-extractor .
   ```

2. **Run the container:**
   ```sh
   docker run --rm -v "$PWD":/app pdf-extractor
   ```
