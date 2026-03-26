# Test the agent with updated reflect() threshold

Write-Host "Testing Agent API with 5-word threshold...`n" -ForegroundColor Cyan

$query = "What is Python?"
$sessionId = "test-session-1"

$body = @{
    query = $query
    session_id = $sessionId
} | ConvertTo-Json

Write-Host "📤 Sending query: $query`n" -ForegroundColor Yellow

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/query" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body -ErrorAction Stop
    
    $result = $response.Content | ConvertFrom-Json
    
    Write-Host "✅ Response received!`n" -ForegroundColor Green
    Write-Host "Answer: $($result.answer)`n" -ForegroundColor Cyan
    Write-Host "Attempts: $($result.attempts)" -ForegroundColor Cyan
    Write-Host "Good: $($result.good)" -ForegroundColor Cyan
    Write-Host "Sources: $($result.sources -join ', ')" -ForegroundColor Cyan
    
    # Check if test passed
    if ($result.attempts -eq 1 -and $result.good -eq $true) {
        Write-Host "`n🎯 TEST PASSED! Attempts: 1, Good: true" -ForegroundColor Green
    } else {
        Write-Host "`n⚠️  TEST RESULT: Attempts: $($result.attempts), Good: $($result.good)" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure the server is running on http://127.0.0.1:8000" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
}
