# Tariff Rule Extractor

Proof-of-concept system that extracts tariff calculation rules from a provided PDF
and turns them into executable Python calculators served through a FastAPI inference API.

This project was completed as an employer-provided technical assignment based on a single PDF.
It is intended to demonstrate an end-to-end approach rather than claim production-level
generalization across arbitrary tariff documents.

## What it does

1. Parses the provided tariff PDF.
2. Retrieves relevant pages for each requested tariff.
3. Extracts calculation rules with LLM assistance.
4. Generates Python calculator functions.
5. Validates results against expected scenarios.
6. Serves accepted calculators through FastAPI.

This project generates and serves tariff calculation rules for port tariff PDFs. The current checked-in inference setup uses rules generated from `pdf_data/Port Tariff.pdf` and exposes them through a FastAPI service.

## Current Test Results

Results below are from `test_report/test_report_20260521_132410.json`. The run used `input_param.json` against the inference endpoint `/ui/infer`.

| Tariff | Expected | Actual | Error |
| --- | ---: | ---: | ---: |
| Light Dues | 60,062.04 | 60,062.04 | 0.00% |
| Port Dues | 199,549.22 | 199,371.3453 | 0.0891% |
| Towage Dues | 147,074.38 | 147,074.38 | 0.00% |
| VTS Dues | 33,315.75 | 33,345.00 | 0.0878% |
| Pilotage Dues | 47,189.94 | 47,189.94 | 0.00% |
| Running Lines | 19,639.50 | 3,805.49 | 80.6233% |

`Running Lines` is included in the registry as the last generated attempt, but it was not accepted as a successful calculator because the result is outside the target tolerance.

## Run the Inference API with Docker

Build and start the API:

```bash
docker compose up --build
```

The service starts on:

```text
http://127.0.0.1:8001
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

Stop the container:

```bash
docker compose down
```

The Docker container runs the Inference API for `Port Tariff.pdf` using the automatically generated document rules already present in the repository:

- `calculators_registry.json`
- `successful_tariff_calculator_*.py`
- `last_tariff_calculator_*.py`
- `pdf_data/Port Tariff.pdf`

The compose file mounts the project directory into `/app`, so the container uses the current local calculator registry, generated calculator files, PDF files, and JSON inputs.

## Use the API

Open the browser UI:

```text
http://127.0.0.1:8001
```

Upload `input_param.json`, select a tariff, and submit the form. The UI calls:

```text
POST /ui/infer
```

The JSON API is also available:

```bash
curl -X POST http://127.0.0.1:8001/infer \
  -H "Content-Type: application/json" \
  -d '{
    "vessel_data": {
      "port": "Durban",
      "technical_specs": {
        "gross_tonnage": 51300,
        "net_tonnage": 31192,
        "loa_meters": 229.2,
        "beam_meters": 38.0,
        "suez_nt": 49069
      },
      "operational_data": {
        "cargo_quantity_mt": 40000,
        "days_alongside": 3.39,
        "arrival_time": "2024-11-15T10:12:00",
        "departure_time": "2024-11-22T13:00:00",
        "activity": "Exporting Iron Ore",
        "num_operations": 2,
        "num_holds": 7
      }
    },
    "target_tariffs": ["Light Dues"],
    "document_name": "Port Tariff.pdf"
  }'
```

For the current project PDF, the browser UI automatically pins the document to `Port Tariff.pdf`.

## Rule Generation Mode

Inference and rule generation are separate modes.

The Docker API is for inference. It loads already generated calculator code from the registry and runs it against vessel input data. It does not need LLM API keys at runtime.

Rule generation is handled by:

```bash
python tariff_langgraph.py
```

To generate rules, create a local `.env` from `.env.example` and fill in the required keys:

```bash
cp .env.example .env
```

Important environment variables:

```env
Gemini_API_KEY=your-gemini-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

TARIFF_RETRIEVAL_MODE=summaries
TARIFF_RAG_TOP_K=5
TARIFF_RAG_DENSE_WEIGHT=0.65

VOYAGE_API_KEY=your-voyage-api-key
VOYAGE_EMBEDDING_MODEL=voyage-3.5-lite
VOYAGE_EMBED_BATCH_SIZE=4
```

To start the generation pipeline, at least one LLM API key must be configured. The first processing round currently uses Anthropic, and the next retrieval/repair round uses Gemini as a fallback path. Providing both keys is recommended, but a run can start when at least one valid key is present.

There are two retrieval modes:

| Mode | Value | Status |
| --- | --- | --- |
| Summaries | `TARIFF_RETRIEVAL_MODE=summaries` | Current working mode. The test results above were produced in this mode. |
| RAG | `TARIFF_RETRIEVAL_MODE=rag` | Example implementation exists, but it still needs fine-tuning. It requires Voyage embeddings configuration. |

Run rule generation for one tariff:

```bash
python tariff_langgraph.py --pdf "pdf_data/Port Tariff.pdf" --tariff "Light Dues"
```

Run several tariffs:

```bash
python tariff_langgraph.py --pdf "pdf_data/Port Tariff.pdf" --tariff "Light Dues" --tariff "Port Dues"
```

Useful logging option:

```bash
python tariff_langgraph.py --pdf "pdf_data/Port Tariff.pdf" --tariff "Towage Dues" --log-file logs/towage_run.log
```
Before running tariff_langgraph.py, delete the following generated files or move them to another directory:
```bash
successful_tariff_calculator_72553f727700e9d_Port_Dues.py
successful_tariff_calculator_af1dc2e1ed809e6_Light_Dues.py
successful_tariff_calculator_b8a7e63a85047be_Pilotage_Dues.py
successful_tariff_calculator_b35b9bfcdc8d1b0_VTS_Dues.py
successful_tariff_calculator_bfc984b6da66bd0_Towage_Dues.py
calculators_registry.json
```
However, deleting these files will break api_service.py. Therefore, it’s better to build the Docker container for api_service.py in advance.

## What `tariff_langgraph.py` Does

The graph builds a calculator by repeatedly finding evidence, extracting rules, writing executable Python, running it, and repairing it when the result is wrong.

```text
START
  |
  v
LoadInputs
  |
  v
ExpandTariffQueries
  |
  +--> RetrieveCandidatePages      when TARIFF_RETRIEVAL_MODE=summaries
  |
  +--> RetrieveCandidatePagesRag   when TARIFF_RETRIEVAL_MODE=rag
  |
  v
InspectPages
  |
  v
EnrichInputData
  |
  v
SelectCalculationParameters
  |
  v
ExtractRules
  |
  v
WriteCode
  |
  v
RunAndSubmit
  |
  v
RepairOrFinish
  |
  +--> WriteCode / InspectPages    if more repair or evidence is needed
  |
  +--> END                         when finished or max rounds reached
```

High-level flow:

1. `LoadInputs` loads the PDF, `input.json`, expected results, existing calculators, pages, and page summaries when summaries mode is enabled.
2. `ExpandTariffQueries` asks the LLM to create search queries for the target tariff.
3. `RetrieveCandidatePages` or `RetrieveCandidatePagesRag` selects pages that likely contain the relevant rules.
4. `InspectPages` extracts evidence from the selected pages.
5. `EnrichInputData` derives additional vessel and operation fields needed by the tariff rules.
6. `SelectCalculationParameters` decides which vessel, cargo, and operational parameters matter.
7. `ExtractRules` converts tariff text into structured calculation logic.
8. `WriteCode` generates a Python calculator with `calculate(vessel_data: dict)`.
9. `RunAndSubmit` executes the generated calculator and evaluates the output.
10. `RepairOrFinish` either saves the successful calculator, tries a code repair, inspects more pages, or stops after the configured limits.

The current generation loop gives the agent two code attempts per tariff. In practice this sometimes fixes the first generated calculator: the first attempt may miss an edge case, then the repair step receives execution/evaluation feedback and writes a corrected version.

The graph also has two processing rounds. The first round uses Anthropic for the initial reasoning and code generation path. If more retrieval or repair is needed, the next round switches to Gemini. This gives the pipeline a second model perspective when the first pass does not produce an accepted calculator.

### Date and Working-Time Enrichment

Tariff rules often depend on the exact moment when an operation happens, not only on vessel size or cargo quantity. During `EnrichInputData`, raw ISO timestamps from the vessel input are expanded into operational calendar fields.

For example, this input timestamp:

```json
{"arrival_time": "2024-11-15T10:12:00"}
```

is enriched into fields like:

```json
{
  "date": "2024-11-15",
  "time": "10:12:00",
  "weekday": "Friday",
  "weekday_number": 5,
  "is_weekend": false,
  "is_standard_workday": true,
  "is_standard_working_hours": true,
  "whole_hours": 10,
  "whole_minutes": 12,
  "hours_until_workday_start": 0,
  "standard_working_hours_assumption": "Monday-Friday 08:00-17:00 local port time"
}
```

`hours_until_workday_start` is calculated against the current local time. When the current local time is already within the standard working window, it is `0`; otherwise it shows how many whole hours remain until the next working day starts at 08:00.

This matters because many port tariffs apply different rates for working hours, after-hours operations, weekends, holidays, night periods, arrival/departure windows, and time alongside.

Many port tariffs depend not only on vessel dimensions or cargo volume, but also on when vessel operations happen. For that reason, the pipeline must extract and preserve date and time fields such as arrival time, departure time, time alongside, operation duration, day/night periods, weekends, holidays, and tariff validity windows whenever the source tariff makes them relevant.
