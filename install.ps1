# Native install on Windows: picks Python 3.12 and runs scripts/install.py.
#   powershell -ExecutionPolicy Bypass -File install.ps1 [install|check] [--accelerator cuda]
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$python = $env:PYTHON
if (-not $python) {
    $python = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
    if (-not $python) { throw 'Python 3.12 not found. Install it (winget install Python.Python.3.12) or set $env:PYTHON.' }
}
& $python scripts/install.py --python $python @args
exit $LASTEXITCODE
