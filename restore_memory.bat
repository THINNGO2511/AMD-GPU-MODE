@echo off
REM Restore Claude Code memory files on Windows.
REM Run after cloning the repo.

set "REPO_DIR=%~dp0"
set "MEMORY_SRC=%REPO_DIR%claude-memory"

REM Claude Code memory path on Windows
REM The path encodes the repo location with dashes
set "CLAUDE_BASE=%USERPROFILE%\.claude\projects"

echo Source: %MEMORY_SRC%
echo.
echo Looking for Claude projects directory...

if not exist "%CLAUDE_BASE%" (
    echo Creating %CLAUDE_BASE%...
    mkdir "%CLAUDE_BASE%"
)

REM Try to find existing project directory
for /d %%D in ("%CLAUDE_BASE%\*AMD-GPU-MODE*") do (
    set "MEMORY_DST=%%D\memory"
    echo Found: %%D
    goto :found
)

REM If not found, create based on common Windows path pattern
set "MEMORY_DST=%CLAUDE_BASE%\AMD-GPU-MODE\memory"
echo No existing project found. Creating: %MEMORY_DST%

:found
if not exist "%MEMORY_DST%" mkdir "%MEMORY_DST%"

echo Copying memory files...
copy "%MEMORY_SRC%\*.md" "%MEMORY_DST%\" >nul

echo.
echo Done! Copied memory files to: %MEMORY_DST%
echo.
echo If Claude Code doesn't pick up the memory, manually copy:
echo   FROM: %MEMORY_SRC%\
echo   TO:   %USERPROFILE%\.claude\projects\[your-project-path]\memory\
echo.
echo Then open Claude Code in this repo and paste PICKUP_PROMPT.md to resume.
pause
