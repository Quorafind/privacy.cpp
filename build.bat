@echo off
setlocal
cmake -S . -B build || exit /b 1
cmake --build build --config Release || exit /b 1
ctest --test-dir build -C Release --output-on-failure || exit /b 1
build\Release\privacy_filter.exe --model model "Alice was born on 1990-01-02." || build\privacy_filter.exe --model model "Alice was born on 1990-01-02."

