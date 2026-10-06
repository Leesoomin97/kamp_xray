param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)

$scriptPath = Join-Path $PSScriptRoot 'build_stage0_data_role_audit.py'
python $scriptPath --workspace $Workspace
