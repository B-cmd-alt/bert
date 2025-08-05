# PowerShell script to monitor and control BERT processes
param(
    [int]$MaxMemoryMB = 4096,
    [int]$MaxCPUPercent = 30
)

function Limit-BertProcesses {
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
    
    foreach ($proc in $pythonProcesses) {
        # Set low priority
        $proc.PriorityClass = "BelowNormal"
        
        # Check memory usage
        $memoryMB = $proc.WorkingSet64 / 1MB
        $cpuPercent = $proc.CPU
        
        Write-Host "Process $($proc.Id): Memory: $([math]::Round($memoryMB))MB"
        
        if ($memoryMB -gt $MaxMemoryMB) {
            Write-Host "Killing process $($proc.Id) - exceeds memory limit"
            Stop-Process -Id $proc.Id -Force
        }
    }
}

# Monitor every 30 seconds
while ($true) {
    Write-Host "$(Get-Date): Monitoring BERT processes..."
    Limit-BertProcesses
    Start-Sleep -Seconds 30
}