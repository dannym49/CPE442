# Compiler
CC = g++

# Output binary name
OUT = main.out

# Compiler flags for warnings and debugging, explicitly add the OpenCV include path
CFLAGS = -std=c++11 -O3 -mcpu=cortex-a72 -fopenmp `pkg-config --cflags opencv4`

# Linker flags to link OpenCV libraries
LDFLAGS = `pkg-config --libs opencv4`

# Source files
SRCS = main.cpp filter.cpp

# Object files (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Target to build everything
all: $(OUT)

# Rule to build the final executable
$(OUT): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(OUT)

# Rule to compile the source files into object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -f $(OBJS) $(OUT)


