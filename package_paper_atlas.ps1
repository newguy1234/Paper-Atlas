$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$dist = Join-Path $root "dist"
$appName = "PaperAtlas-Portable-$stamp"
$stage = Join-Path $dist $appName
$package = Join-Path $stage "paper_atlas"
$zipPath = Join-Path $dist "$appName.zip"

New-Item -ItemType Directory -Force -Path $package | Out-Null
Get-ChildItem -LiteralPath (Join-Path $root "paper_atlas") -Filter "*.py" -File |
    ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $package }
Copy-Item -LiteralPath (Join-Path $root "README_PAPER_ATLAS.md") -Destination $stage
Copy-Item -LiteralPath (Join-Path $root "requirements-paper-atlas.txt") -Destination $stage
Copy-Item -LiteralPath (Join-Path $root "run_paper_atlas.ps1") -Destination $stage
Copy-Item -LiteralPath (Join-Path $root "start_paper_atlas.bat") -Destination $stage
Copy-Item -LiteralPath (Join-Path $root "package_paper_atlas.ps1") -Destination $stage

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Portable app folder: $stage"
Write-Host "Portable app zip:    $zipPath"
