python setup.py clean --all
@setlocal
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
set CXXFLAGS=-march=native
set NATIVE_UCI=1
python setup.py build_ext --inplace
@endlocal