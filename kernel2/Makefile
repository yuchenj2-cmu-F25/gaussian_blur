# ------------------------------------------------------------
# Makefile for Kernel 2 (Sum of Squares) Benchmark
# ------------------------------------------------------------

CC = gcc
CFLAGS = -O3 -march=broadwell -mavx2 -mfma -std=c11 -Wall -Wextra -fopenmp
TARGET = kernel2_bench
SRC = sum_of_squares_main.c sum_of_squares.c utils.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

