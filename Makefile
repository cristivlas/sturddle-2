# Makefile for standalone executable

CC = clang-18
CXX = clang++-18

SOURCES = chess.cpp context.cpp search.cpp uci_native.cpp main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

DEPS = $(OBJECTS:.o=.d) $(MAIN_OBJECT:.o=.d)

# Target executable
TARGET = sturddle-2.3

# Include directories
INCLUDES = -I./libpopcnt -I./magic-bits/include -I./version2 -I/usr/include/python3.11

# Build timestamp
BUILD_STAMP = $(shell date +%m%d%y)

CXXFLAGS = -MMD -MP \
        -O3 \
        -std=c++20 \
        -stdlib=libc++ \
        -fexperimental-library \
        -DNO_ASSERT \
        -D_FORTIFY_SOURCE=0 \
        -DSTANDALONE \
        -DUSE_MAGIC_BITS=false \
        -DBUILD_STAMP=$(BUILD_STAMP) \
        -DCALLBACK_PERIOD=8192 \
        -fno-stack-protector \
        -DWITH_NNUE \
        -DNATIVE_UCI=true \
        -DUSE_MAGIC_BITS=false \
        -Wextra \
        -Wno-unused-label \
        -Wno-unknown-pragmas \
        -Wno-unused-parameter \
        -Wno-unused-variable \
        -Wno-empty-body \
        -Wno-int-in-bool-context \
        -Wno-macro-redefined \
        -Wno-deprecated-declarations \
        -march=native \
        $(INCLUDES)

#. Linker flags
LDFLAGS = \
        -fuse-ld=lld \
        -L/usr/lib/llvm-18/lib/ \
        -L/usr/lib/llvm-18/lib/x86_64-pc-linux-gnu \
        -lc++ \
        -lc++experimental


# Default target
all: $(TARGET)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(OBJECTS) $(MAIN_OBJECT)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(MAIN_OBJECT) -o $(TARGET)

# Clean build files
clean:
	rm -f $(OBJECTS) $(MAIN_OBJECT) $(TARGET) $(DEPS)

# Print variables for debugging
debug:
	@echo "CC: $(CC)"
	@echo "CXX: $(CXX)"
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"

.PHONY: all run clean debug

-include $(DEPS)
