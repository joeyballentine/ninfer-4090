<#
.SYNOPSIS
Start ninfer-serve on a 24 GB RTX 4090 with one of the measured long-context profiles.

.DESCRIPTION
Each profile is a KV mode and context size verified on an RTX 4090 with MTP speculation; the
README's "Long context" section has the measurements. Arguments after the named parameters pass
through to ninfer-serve unchanged, so later flags override the profile.

.EXAMPLE
tools\serve_4090.ps1
tools\serve_4090.ps1 -Context 256k -Port 8081
tools\serve_4090.ps1 -Context 128k --api-key secret --cors
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    # 128k: rk8v4, +0.09% code ppl. 160k and 224k: rk4v4-e8, +0.32%. 256k: rk2v4-e8, +3.1%.
    [ValidateSet('128k', '160k', '224k', '256k')]
    [string]$Context = '160k',
    [string]$Model = 'models\qwen3_8_27b_24gb.ninfer',
    [int]$Port = 8080,
    # Print the command without starting the server.
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ServeArgs = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build\apps\ninfer-serve.exe'
$modelPath = if ([System.IO.Path]::IsPathRooted($Model)) { $Model } else { Join-Path $root $Model }

if (-not (Test-Path $exe)) { throw "ninfer-serve not built: $exe. Build with -DCMAKE_CUDA_ARCHITECTURES=89." }
if (-not (Test-Path $modelPath)) { throw "model not found: $modelPath" }

$profiles = @{
    '128k' = @('--kv-dtype', 'rk8v4', '--max-context', '131072', '--kv-capacity', '131072')
    '160k' = @('--kv-dtype', 'rk4v4-e8', '--max-context', '163840', '--kv-capacity', '163840')
    # The two larger windows need the cached device checkpoint dropped and a smaller prefill
    # workspace to keep about 1 GiB free.
    '224k' = @('--kv-dtype', 'rk4v4-e8', '--max-context', '229376', '--kv-capacity', '229376',
               '--device-state-slots', '0', '--prefill-chunk', '1024')
    '256k' = @('--kv-dtype', 'rk2v4-e8', '--max-context', '262144', '--kv-capacity', '262144',
               '--device-state-slots', '0', '--prefill-chunk', '1024')
}

# MTP at K=5 is the fastest measured draft window. The default 8 GiB pinned host cache only serves
# switching between conversations, so it is cut to 1 GiB.
$common = @('--spec', 'mtp', '--draft-tokens', '5', '--lm-head-draft',
            '--host-kv-mib', '1024', '--host-state-slots', '2', '--port', "$Port")

$argv = @($modelPath) + $profiles[$Context] + $common + $ServeArgs
Write-Host "$exe $($argv -join ' ')"
if ($DryRun) { return }

& $exe @argv
exit $LASTEXITCODE
