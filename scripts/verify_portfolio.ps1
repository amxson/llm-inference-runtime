param([string]$OutputDirectory = 'docs/evidence/final/checks')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$evidenceOutput = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
if (-not $evidenceOutput.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Evidence output must stay in the repository.' }
New-Item -ItemType Directory -Path $evidenceOutput -Force | Out-Null
$nodeCommand = (Get-Command node -ErrorAction Stop).Source
$pythonCommand = Join-Path $repoRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $pythonCommand)) { $pythonCommand = (Get-Command python -ErrorAction Stop).Source }
$env:NO_COLOR = '1'
$checkResults = @()
function Invoke-Check {
  param([string]$Name,[string]$Program,[string[]]$Arguments,[string]$DisplayCommand)
  $runStart = Get-Date -AsUTC -Format o
  $lines = @(& $Program @Arguments 2>&1)
  $runExit = $LASTEXITCODE
  $clean = ($lines | ForEach-Object { "$($_)".Replace($repoRoot,'<repo>').Replace($env:USERPROFILE,'<user>').Replace($repoRoot.Replace('\','/'),'<repo>').Replace($env:USERPROFILE.Replace('\','/'),'<user>') }) -join "`n"
  $clean = (($clean -split "`n" | ForEach-Object { $_.TrimEnd() }) -join "`n").TrimEnd()
  [System.IO.File]::WriteAllText((Join-Path $evidenceOutput "$Name.txt"),$(if ($clean) { $clean + "`n" } else { '' }))
  Write-Output "$Name : exit $runExit"
  $script:checkResults += [pscustomobject]@{name=$Name;utc=$runStart;command=$DisplayCommand;exit_code=$runExit}
  $script:checkResults | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $evidenceOutput 'results.json') -Encoding utf8
  if ($runExit -ne 0) { Write-Output $clean; throw "$Name failed" }
}
Set-Location $repoRoot
Invoke-Check 'python-tests' $pythonCommand @('-m','pytest','-q') 'python -m pytest -q'
Invoke-Check 'python-compile' $pythonCommand @('-m','compileall','-q','src','scripts','tests') 'python -m compileall -q src scripts tests'
Invoke-Check 'git-diff-check' 'git' @('diff','--check') 'git diff --check'
Set-Location (Join-Path $repoRoot 'frontend')
Invoke-Check 'benchmark-sync' $nodeCommand @('scripts/sync-data.mjs') 'node scripts/sync-data.mjs'
Invoke-Check 'frontend-tests' $nodeCommand @('node_modules/vitest/vitest.mjs','run') 'vitest run'
Invoke-Check 'typecheck' $nodeCommand @('node_modules/typescript/bin/tsc','--noEmit') 'tsc --noEmit'
Invoke-Check 'design-audit' $nodeCommand @('scripts/design-audit.mjs') 'node scripts/design-audit.mjs'
Invoke-Check 'production-build' $nodeCommand @('node_modules/vite/bin/vite.js','build') 'vite build'
Set-Location $repoRoot
