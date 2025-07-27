# Build with your proven approach
docker build -t persona-doc-analyzer .

# Run the container
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  persona-doc-analyzer \
  --input /app/input/input.json \
  --output /app/output/output.json