python setup.py clean --all
set CL_EXE=C:\Program Files\LLVM\bin\clang-cl.exe
set CFLAGS="-march=native %CFLAGS%"
set NATIVE_UCI=1
python setup.py build_ext --inplace
