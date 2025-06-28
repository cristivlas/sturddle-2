python setup.py clean --all
@setlocal
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
@REM set CXXFLAGS=-march=native -DUSE_MMAP_HASH_TABLE
set CXXFLAGS=-march=native
set NATIVE_UCI=1
python setup.py build_ext --inplace
@endlocal
