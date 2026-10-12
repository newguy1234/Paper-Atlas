# Paper Atlas

Paper Atlas is a local Solara research app for finding current papers, collecting open-access PDFs, extracting methods and claims with source quotes, asking questions across a saved collection, and exploring how the papers connect.

## Start the app on Windows

From this folder, double-click `start_paper_atlas.bat`, or run:

```powershell
.\run_paper_atlas.ps1
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765). The first launch creates a separate `.venv-paper-atlas` environment and installs the app dependencies. Python 3.11 or newer and internet access are required for first-time setup. Stop the app with Ctrl+C in that PowerShell window.

## In the app

1. Search OpenAlex, arXiv, and Semantic Scholar. Choose either a strict computer-science methods scope (OpenAlex primary CS field, arXiv CS categories, and Semantic Scholar Computer Science field) or a broader cross-field search for CS methods applied in other disciplines. Each index can be enabled or disabled, and results are merged and ranked into a local collection.
2. Paste a BibTeX bibliography to import its records into a separate local collection.
3. Select one, several, or all papers. Choose any local folder for downloaded PDFs; if you select a OneDrive-synced folder, OneDrive uploads them when its desktop client is running. The app detects common OneDrive locations automatically and remembers your selected folder.
4. Download only provider-listed open-access PDFs, or scrape selected papers for methods and claims. Long-running work happens in the background while the page remains usable.
5. Open the evidence map. Provider citation records, shared topics, quote-checked extracted facts, and model-inferred relationships are shown as different link types.
6. Ask a question over the saved collection using local Ollama. Change the model or Ollama address in the UI.
7. Export a collection as JSON (including graph/facts), CSV, or BibTeX.

The library is stored in `data/paper_atlas/library.sqlite3`; the default file destination is your Downloads folder, unless you choose another folder such as OneDrive in the app. The database remains local. PDF fetching is limited to HTTPS provider-listed open-access links and public arXiv PDFs. A PDF is limited to 35 MB, each batch to 250 MB, checked for a PDF signature, and never fetched until you choose download or extraction. If a PDF cannot be reached, extraction can fall back to the abstract and reports that source downgrade.

## Local Ollama

Install and start Ollama separately, then make the chosen model available there. The app defaults to `olmo-3:7b-instruct` at `http://127.0.0.1:11434`; both values can be changed in the app. Search and graph browsing do not require Ollama.

## Use from an MCP client

Paper Atlas also exposes local tools for search, graph inspection, PDF download, quote-checked extraction, and collection Q&A. After the app environment has been created, configure your MCP client to run this command from the repository folder:

```powershell
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.mcp_server
```

The server uses stdio by default. Its database path can be changed with `PAPER_ATLAS_DB`. Keep the working directory at the project root so Python can import the package. The existing `mcp_server.py` for the QASA corpus is untouched.

## Command line

```powershell
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli search "agentic scientific RAG" --limit 20 --from-year 2023
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli research-pack --preset qasa --scope cs_methods --out results\qasa_papers.json
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli research-pack --preset qasa_transfer --scope cs_applied --providers openalex arxiv --out results\qasa_transfer_papers.json
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli download PAPER_ID --out-dir "$env:OneDriveCommercial\Paper Atlas\PDFs"
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli read PAPER_ID --pdf-dir "$env:OneDriveCommercial\Paper Atlas\PDFs" --out results\paper_text.txt
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli analyze PAPER_ID --pdf-dir "$env:OneDriveCommercial\Paper Atlas\PDFs"
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli bibtex .\references.bib --name "My BibTeX papers"
.\.venv-paper-atlas\Scripts\python.exe -m paper_atlas.cli graph COLLECTION_ID --out graph.json
```

`research-pack` runs a set of targeted searches, merges duplicate records with reciprocal-rank fusion, and saves one local collection. The built-in `qasa` preset focuses on computer-science methods; `qasa_transfer` looks for computer-science retrieval and reasoning methods used in scientific QA systems across biology, medicine, chemistry, physics, and engineering, including accelerator science. It targets both methods and their evaluations, so results can suggest transfer experiments without assuming that gains transfer to QASA. Use `--scope cs_applied` for these cross-field applications; the strict `cs_methods` scope intentionally filters for CS as the primary field. Repeat `--query "..."` to use your own query set instead. `bibtex` imports a `.bib` file from the terminal; the app also accepts pasted BibTeX.

`download` saves provider-listed open PDFs. `read` extracts page-numbered full text to UTF-8 without using a local model, which supports local text analysis while another workload is running. `analyze` optionally uses Ollama to extract methods and claims, and validates every returned quote against the PDF text. For a personal OneDrive sync folder, substitute its full synced path if `$env:OneDriveCommercial` is not set.

Provider metadata is retrieved from OpenAlex, arXiv, and Semantic Scholar. Optional API keys are read from `OPENALEX_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`; the app works without them subject to provider rate limits. The packaged app includes the Solara interface and the terminal command line, but creates its own Python environment on first launch.

## Build the portable Windows app

Run `package_paper_atlas.ps1`. It creates a dated folder and zip under `dist/`. Extract the zip, then double-click `start_paper_atlas.bat`. The launcher creates an isolated environment on first run, starts the local app, and opens the browser when it is ready. The app binds to `127.0.0.1` and does not expose its library to the network.

## Save location

The Save location card is available before your first search. Paste a folder path and click **Save folder**, or click **Use Downloads** to reset it. Blank input uses Downloads. Windows redirected Downloads folders are detected. Existing saved preferences are retained. The same remembered folder is used by UI, CLI and MCP PDF downloads and extraction; an explicit CLI folder overrides it for that command.

**Save exports to folder** writes JSON, CSV and BibTeX into that location. Re-exporting the same collection updates those three files. Browser export buttons still use your browser download preferences. The library database remains in the app's local data directory; changing the save location does not move existing files.

## Source package

This repository contains app source and launchers, not a compiled executable. No downloaded papers, library database, API keys, model weights or competition artifacts are included. Search, BibTeX import and exports work without Ollama; model-assisted extraction and questions require your own running Ollama installation.
