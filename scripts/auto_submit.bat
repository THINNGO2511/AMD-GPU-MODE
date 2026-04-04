@echo off
REM Auto-submit loop for Windows. Submits best submissions to leaderboard every hour.
REM Run with: start auto_submit.bat

cd /d "%~dp0"

:loop
echo === %date% %time% === AUTO SUBMIT ===

REM GEMM
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/submission_stages3_all.py --no-tui 2>&1 | findstr /i "Rate limit" >nul && (echo GEMM: rate limited) || (echo GEMM: submitted)

REM MLA
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/submission_pg2_pingpong.py --no-tui 2>&1 | findstr /i "Rate limit" >nul && (echo MLA: rate limited) || (echo MLA: submitted)

REM MoE
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_envvars_moe.py --no-tui 2>&1 | findstr /i "Rate limit" >nul && (echo MoE: rate limited) || (echo MoE: submitted)

echo Sleeping 3600s (1hr)...
timeout /t 3600 /nobreak >nul
goto loop
