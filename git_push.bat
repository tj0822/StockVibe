@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Git add/commit/push 자동화 스크립트

cd /d "%~dp0"

REM 커밋 메시지 입력
if "%1"=="" (
    set "COMMIT_MSG=StockVibe 업데이트"
) else (
    set "COMMIT_MSG=%*"
)

echo.
echo ========================================
echo Git 작업 시작
echo ========================================
echo 커밋 메시지: %COMMIT_MSG%
echo.

REM git status 확인
echo [1/4] 현재 상태 확인...
git status
echo.

REM git add
echo [2/4] 모든 변경사항 스테이징...
git add .
if errorlevel 1 (
    echo ❌ git add 실패
    exit /b 1
)
echo ✅ git add 완료
echo.

REM git commit
echo [3/4] 변경사항 커밋...
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo ❌ git commit 실패
    exit /b 1
)
echo ✅ git commit 완료
echo.

REM git push
echo [4/4] 원격 저장소에 푸시...
git push
if errorlevel 1 (
    echo ❌ git push 실패
    exit /b 1
)
echo ✅ git push 완료
echo.

echo ========================================
echo ✅ 모든 작업 완료!
echo ========================================
pause
