@echo off
setlocal

if "%CC%"=="" set CC=clang-cl

set CMD=%CC% /std:c++20 /W4 /EHsc /I.. /I..\libpopcnt /I..\magic-bits\include -Wno-unused-parameter -Wno-unused-variable codegen.cpp ..\chess.cpp /Fe:codegen.exe

echo %CMD%
%CMD%

endlocal
