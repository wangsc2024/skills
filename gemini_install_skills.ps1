$skillsSourcePath = "d:\Source\skills"
$targetPath = "$HOME\.gemini\skills"
$skillsIndexPath = Join-Path $skillsSourcePath "SKILLS_INDEX.md"

if (-not (Test-Path $targetPath)) {
    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
}

$content = Get-Content $skillsIndexPath -Raw

# 1. 處理目錄型 Skill (從表格中擷取)
Write-Host "正在處理目錄型 Skills..." -ForegroundColor Cyan
$directorySkills = [regex]::Matches($content, '\| \*\*(\w+[\w-]*)\*\* \| .*? \| .*? \| 目錄.*? \|') | ForEach-Object { $_.Groups[1].Value }

foreach ($skill in $directorySkills) {
    $src = Join-Path $skillsSourcePath $skill
    $dest = Join-Path $targetPath $skill
    
    if (Test-Path $src) {
        Write-Host "安裝目錄型 Skill: $skill"
        Copy-Item -Path $src -Destination $dest -Recurse -Force
    } else {
        Write-Warning "找不到 Skill 目錄: $src"
    }
}

# 2. 處理指令型 Skill (從指令型 Skills 區段擷取)
Write-Host "`n正在處理指令型 Skills..." -ForegroundColor Cyan
# 匹配 | ai_daily_report_instruction.md | ai-daily-report | ... |
$instructionPattern = '\| ([\w-]+\.md) \| ([\w-]+) \|'
$instructionMatches = [regex]::Matches($content, $instructionPattern)

foreach ($match in $instructionMatches) {
    $fileName = $match.Groups[1].Value
    $skillName = $match.Groups[2].Value
    
    $srcFile = Join-Path $skillsSourcePath $fileName
    $destFolder = Join-Path $targetPath $skillName
    $destFile = Join-Path $destFolder "SKILL.md"
    
    if (Test-Path $srcFile) {
        Write-Host "安裝指令型 Skill: $skillName (來源: $fileName)"
        if (-not (Test-Path $destFolder)) {
            New-Item -ItemType Directory -Path $destFolder -Force | Out-Null
        }
        Copy-Item -Path $srcFile -Destination $destFile -Force
    } else {
        Write-Warning "找不到指令檔: $srcFile"
    }
}

Write-Host "`nSkill 安裝完成！" -ForegroundColor Green
