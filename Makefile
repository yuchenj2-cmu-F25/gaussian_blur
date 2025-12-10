# ------------------------------------------------------------
# Makefile for Gaussian Blur Benchmark
# ------------------------------------------------------------

CC      := gcc
CFLAGS  := -O3 -march=native -Wall -Wextra -fopenmp -Ikernel1 -Ikernel2
TARGET  := blur_test
# Exclude kernel2/sum_of_squares_main.c (standalone benchmark) and kernel2/utils.c (duplicates kernel1/utils.c)
KERNEL2_SRCS := $(filter-out kernel2/sum_of_squares_main.c kernel2/utils.c,$(wildcard kernel2/*.c))
SRCS    := main.c $(wildcard kernel1/*.c) $(KERNEL2_SRCS)
OBJS    := $(SRCS:.c=.o)
HEADERS := $(wildcard kernel1/*.h) $(wildcard kernel2/*.h)

# Default build rule
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Build object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Remove build artifacts
clean:
	rm -f $(TARGET) $(OBJS)

# Optional: rebuild from scratch
rebuild: clean all

.PHONY: all run clean rebuild
