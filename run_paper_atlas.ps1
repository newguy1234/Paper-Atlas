$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$root$([IO.Path]::PathSeparator)$env:PYTHONPATH" } else { $root }
$venv = Join-Path $root ".venv-paper-atlas"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    $basePython = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $basePython) {
        & $basePython -m venv $venv
    } else {
        $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if (-not $pyLauncher) {
            throw "Python was not found. Install Python 3.11+ or use the repository's .venv, then start Paper Atlas again."
        }
        & $pyLauncher.Source -3 -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Paper Atlas Python environment." }
}

$probe = & $python -c "import importlib.util; names=['solara','networkx','matplotlib','fitz','requests','mcp']; print('ready' if all(importlib.util.find_spec(n) for n in names) else 'missing')"
if ($LASTEXITCODE -ne 0 -or $probe -notcontains "ready") {
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update pip in the Paper Atlas environment." }
    & $python -m pip install -r (Join-Path $root "requirements-paper-atlas.txt")
    if ($LASTEXITCODE -ne 0) { throw "Paper Atlas dependencies could not be installed." }
}

Write-Host "Paper Atlas is local at http://127.0.0.1:8765"
Write-Host "Your browser will open when the page is ready. Press Ctrl+C here to stop the app."
$url = "http://127.0.0.1:8765/"
$browserJob = Start-Job -ArgumentList $url -ScriptBlock {
    param($targetUrl)
    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $targetUrl -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Start-Process $targetUrl
                return
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
}
try {
    & $python -m solara run (Join-Path $root "paper_atlas\app.py") --host 127.0.0.1 --port 8765
} finally {
    if ($browserJob) {
        Stop-Job $browserJob -ErrorAction SilentlyContinue
        Remove-Job $browserJob -Force -ErrorAction SilentlyContinue
    }
}
