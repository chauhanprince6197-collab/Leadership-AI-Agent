# ============================================================
#  setup_project.ps1
#  Run this ONCE from the folder where all your .py files are.
#  It creates the correct project structure automatically.
#
#  Usage:
#    1. Put all your downloaded files in one folder (e.g. C:\assignment)
#    2. Open PowerShell in that folder
#    3. Run:  .\setup_project.ps1
# ============================================================

Write-Host ""
Write-Host "=== AI Leadership Agent — Project Setup ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. Create all required folders ──────────────────────────────────────────
$folders = @(
    "app",
    "app\api",
    "app\generation",
    "app\ingestion",
    "app\retrieval",
    "config",
    "frontend",
    "scripts",
    "tests",
    "documents"
)

foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f | Out-Null
        Write-Host "  Created  $f\" -ForegroundColor Green
    } else {
        Write-Host "  Exists   $f\" -ForegroundColor Gray
    }
}

# ── 2. Create all __init__.py files ─────────────────────────────────────────
$initFiles = @(
    "app\__init__.py",
    "app\api\__init__.py",
    "app\generation\__init__.py",
    "app\ingestion\__init__.py",
    "app\retrieval\__init__.py",
    "config\__init__.py",
    "tests\__init__.py"
)

# The main app\__init__.py comes from the uploaded __init__.py
if (Test-Path "__init__.py") {
    Copy-Item "__init__.py" "app\__init__.py" -Force
    Write-Host "  Copied   __init__.py  ->  app\__init__.py" -ForegroundColor Yellow
}

# Empty __init__.py for other packages
foreach ($f in $initFiles) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType File -Path $f | Out-Null
        Write-Host "  Created  $f" -ForegroundColor Green
    }
}

# ── 3. Move each file to its correct location ────────────────────────────────
$moves = @{
    # source filename        = destination path
    "pipeline.py"            = "app\ingestion\pipeline.py"
    "store.py"               = "app\retrieval\store.py"
    "router.py"              = "app\retrieval\router.py"
    "chain.py"               = "app\generation\chain.py"
    "routes.py"              = "app\api\routes.py"
    "settings.py"            = "config\settings.py"
    "ingest.py"              = "scripts\ingest.py"
    "index.html"             = "frontend\index.html"
    "conftest.py"            = "tests\conftest.py"
    "test_pipeline.py"       = "tests\test_pipeline.py"
    # root-level files stay in place
    "main.py"                = "main.py"
    "requirements.txt"       = "requirements.txt"
    "Dockerfile"             = "Dockerfile"
    "docker-compose.yml"     = "docker-compose.yml"
}

foreach ($src in $moves.Keys) {
    $dst = $moves[$src]
    if (Test-Path $src) {
        # Don't overwrite destination if it's the same file
        if ($src -ne $dst) {
            Copy-Item $src $dst -Force
            Write-Host "  Moved    $src  ->  $dst" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  MISSING  $src  (skipped)" -ForegroundColor Red
    }
}

# ── 4. Handle the .env files (downloaded as _env / _env.example) ─────────────
if (Test-Path "_env") {
    Copy-Item "_env" ".env" -Force
    Write-Host "  Copied   _env  ->  .env" -ForegroundColor Yellow
}
if (Test-Path "_env.example") {
    Copy-Item "_env.example" ".env.example" -Force
    Write-Host "  Copied   _env.example  ->  .env.example" -ForegroundColor Yellow
}

# ── 5. Show final structure ──────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Final structure ===" -ForegroundColor Cyan
Get-ChildItem -Recurse -Depth 3 |
    Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.FullName -notmatch '\\chroma_db\\' } |
    ForEach-Object {
        $indent = "  " * ($_.FullName.Split("\").Count - $PWD.FullName.Split("\").Count)
        if ($_.PSIsContainer) {
            Write-Host "$indent$($_.Name)\" -ForegroundColor Cyan
        } else {
            Write-Host "$indent$($_.Name)" -ForegroundColor White
        }
    }

# ── 6. Check .env has API key ────────────────────────────────────────────────
Write-Host ""
if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "sk-ant-") {
        Write-Host "✓ .env found with API key" -ForegroundColor Green
    } else {
        Write-Host "⚠  .env exists but ANTHROPIC_API_KEY not set yet" -ForegroundColor Yellow
        Write-Host "   Open .env in Notepad and add:" -ForegroundColor Yellow
        Write-Host "   ANTHROPIC_API_KEY=sk-ant-your-key-here" -ForegroundColor White
    }
} else {
    Write-Host "⚠  No .env file found. Creating one..." -ForegroundColor Yellow
    "ANTHROPIC_API_KEY=sk-ant-your-key-here" | Out-File -FilePath ".env" -Encoding utf8
    Write-Host "   Edit .env and replace sk-ant-your-key-here with your real key" -ForegroundColor White
}

# ── 7. Next steps ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Next steps ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. Install dependencies:"
Write-Host "     pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "  2. Put your .txt files in the documents\ folder"
Write-Host ""
Write-Host "  3. Ingest documents:"
Write-Host "     python scripts\ingest.py --folder .\documents" -ForegroundColor White
Write-Host ""
Write-Host "  4. Start the server:"
Write-Host "     python main.py" -ForegroundColor White
Write-Host ""
Write-Host "  5. Open browser: http://localhost:5050"
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
