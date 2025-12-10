# Makefile for Kernel 3: Direction Classification Benchmark

CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -ffast-math
LDFLAGS := -lm

# Output
TARGET := kernel_benchmark
SRCS := kernel_benchmark.c kernel_naive.c kernel_basic_avx2.c kernel_integer_opt.c kernel_reduction_tree.c
OBJS := $(SRCS:.c=.o)
HEADERS := direction_kernels.h

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

rebuild: clean all

# Helpful targets
analyze: $(TARGET)
	@echo "Running detailed benchmark analysis..."
	./$(TARGET)

.PHONY: all run clean rebuild analyze
