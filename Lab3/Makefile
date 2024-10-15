#Compiler
CC = g++ 

#Output file
OUT = main.out 

#Linker flags
LDFLAGS = -g 'pkg-config --cflags opencv4'

#Compiler Flags
CFLAGS = -Werror -g 'pkg-config --cflags opencv4'

#Header Files
DEPS = grayscale.hpp sobel.hpp

#Source Files
SRCS = grayscale.cpp sobel.cpp main.cpp

OBJS = $(SRCS:.cpp=-o)

#Default target
all: ${OUT}

#Rule to compile .cpp files to .o object files
%.o: %.cpp ${DEPS}
	${CC} -c ${CFLAGS} $< -o $@

#Clean up the build
clean:
		rm -f ${OUT}