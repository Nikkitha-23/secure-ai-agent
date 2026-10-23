# PowerShell script to create Desktop shortcut for Secure AI Agent Chatbot
$projectDir = Split-Path -Path $PSScriptRoot -Parent
$targetBat = Join-Path -Path $projectDir -ChildPath "launch_chatbot.bat"
$desktopDir = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$shortcutPath = Join-Path -Path $desktopDir -ChildPath "Secure AI Agent.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $targetBat
$Shortcut.WorkingDirectory = $projectDir
$Shortcut.Description = "Single-Click Launcher for Secure AI Agent Chatbot"
$Shortcut.Save()

Write-Host "✅ Desktop Shortcut created successfully at:" -ForegroundColor Green
Write-Host "   $shortcutPath" -ForegroundColor Cyan
