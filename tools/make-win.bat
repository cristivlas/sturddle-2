python setup.py clean --all
@setlocal
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
set CXXFLAGS=-march=native -DSHARED_WEIGHTS -Wmissing-field-initializers -Werror
@REM set CXXFLAGS=%CXXFLAGS% -DUSE_MMAP_HASH_TABLE
python setup.py build_ext --inplace
@endlocal
