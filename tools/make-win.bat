python setup.py clean --all
@setlocal
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
if not defined MARCH set MARCH=-march=native
set CXXFLAGS=%CXXFLAGS% %MARCH% -Wmissing-field-initializers -Werror
set CXXFLAGS=%CXXFLAGS% -DSHARED_WEIGHTS
@REM set CXXFLAGS=%CXXFLAGS% -DUSE_MMAP_HASH_TABLE
python setup.py build_ext --inplace
@endlocal
