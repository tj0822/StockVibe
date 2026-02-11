@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM Git add/commit/push 자동화 스크립트

cd /d "%~dp0"

echo.
echo ========================================
echo Git 작업 시작
echo ========================================
echo.

REM git status 확인
echo [1/5] 현재 상태 확인...
git status --short
echo.

REM 변경된 파일 목록 수집
echo [2/5] 변경 내역 분석...
set "CHANGED_FILES="
set "FILE_COUNT=0"

for /f "tokens=1,2 delims= " %%a in ('git status --short 2^>nul') do (
    set /a FILE_COUNT+=1
    if !FILE_COUNT! leq 5 (
        if "!CHANGED_FILES!"=="" (
            set "CHANGED_FILES=%%b"
        ) else (
            set "CHANGED_FILES=!CHANGED_FILES!, %%b"
        )
    )
)

if %FILE_COUNT% gtr 5 (
    set "CHANGED_FILES=!CHANGED_FILES! 외 %FILE_COUNT%개 파일"
)

echo 변경된 파일: !CHANGED_FILES!
echo.

REM 오늘 날짜
for /f "tokens=1-3 delims=/" %%a in ('echo %date%') do (
    set "TODAY=%%a-%%b-%%c"
)

REM 커밋 메시지 생성
if "%1"=="" (
    set "AUTO_MSG=[%TODAY%] !CHANGED_FILES! 업데이트"
    echo 자동 생성된 커밋 메시지:
    echo   !AUTO_MSG!
    echo.
    set /p "USER_INPUT=추가 설명 (Enter: 자동 메시지 사용): "
    if "!USER_INPUT!"=="" (
        set "COMMIT_MSG=!AUTO_MSG!"
    ) else (
        set "COMMIT_MSG=[%TODAY%] !USER_INPUT!"
    )
) else (
    set "COMMIT_MSG=[%TODAY%] %*"
)

echo.
echo 최종 커밋 메시지: !COMMIT_MSG!
echo.

REM git add
echo [3/5] 모든 변경사항 스테이징...
git add .
if errorlevel 1 (
    echo ❌ git add 실패
    exit /b 1
)
echo ✅ git add 완료
echo.

REM git commit
echo [4/5] 변경사항 커밋...
git commit -m "!COMMIT_MSG!"
if errorlevel 1 (
    echo ❌ git commit 실패
    exit /b 1
)
echo ✅ git commit 완료
echo.

REM git push
echo [5/5] 원격 저장소에 푸시...
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
