CC = g++

OUT = main.out 

LDFLAGS = -g

CFLAGS = -Werror pkg-config --cflags opencv4

DEPS = grayscale.hpp sobel.hpp

SRCS = grayscale.cpp sobel.cpp main.cpp

all: default

default: 
		${CC} ${CFLAGS} ${LDFLAGS} ${SRCS} ${DEPS} -o ${OUT}

clean:
		rm -f ${OUT}