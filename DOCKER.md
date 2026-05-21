# Tariff Inference API Docker

Build and start the API:

```powershell
docker compose up --build
```

Open the UI:

```text
http://127.0.0.1:8001
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/health
```

Run the existing smoke test against the container:

```powershell
.\.venv\Scripts\python.exe test_api_inference.py --api http://127.0.0.1:8001
```

Stop the container:

```powershell
docker compose down
```

The compose file mounts the project into `/app`, so the container uses the current
`calculators_registry.json`, generated calculator modules, PDF files, and input JSON files.
The local `.env` file is passed at runtime but is excluded from the Docker image.
