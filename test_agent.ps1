# Test the agent with updated reflect() threshold

Write-Host "Testing Agent API with 5-word threshold..." -ForegroundColor Cyan
Write-Host ""

`$query = "What is Python?"
`$sessionId = "test-session-1"

`$body = @{
    query = `$query
    session_id = `$sessionId
} | ConvertTo-Json

Write-Host "Sending query: `$query" -ForegroundColor Yellow
Write-Host ""

try {
    `$response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/query" `
        -Method POST `
        -ContentType "application/json" `
        -Body `$body -ErrorAction Stop
    
    `$result = `$response.Content | ConvertFrom-Json
    
    Write-Host "Response received!" -ForegroundColor Green
    Write-Host "Answer: `$(`$result.answer)" -ForegroundColor Cyan
    Write-Host "Attempts: `$(`$result.attempts)" -ForegroundColor Cyan
    Write-Host "Good: `$(`$result.good)" -ForegroundColor Cyan
    
    if (`$result.attempts -eq 1 -and `$result.good -eq `$true) {
        Write-Host "TEST PASSED! Attempts: 1, Good: true" -ForegroundColor Green
    } else {
        Write-Host "TEST RESULT: Attempts: `$(`$result.attempts), Good: `$(`$result.good)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "Error: `$(`$_.Exception.Message)" -ForegroundColor Red
}
