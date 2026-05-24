# Start Kokomi Chat - PowerShell version
# Automatically activates venv and launches chat

Write-Host "`n╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     KOKOMI - Local AI Assistant                      ║" -ForegroundColor Cyan
Write-Host "║     Starting chat interface...                        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate venv
& "$scriptDir\venv\Scripts\Activate.ps1"

# Run chat
Write-Host "Model loading..." -ForegroundColor Yellow
& python main.py

Write-Host "`nChat ended. Press any key to close." -ForegroundColor Gray
$null = [Console]::ReadKey($true)
