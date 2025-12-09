# ------------------------------------------------------------
# Makefile for Gaussian Blur Benchmark
# ------------------------------------------------------------

CC      := gcc
CFLAGS  := -O3 -march=native -Wall -Wextra -Ikernel1
TARGET  := blur_test
SRCS    := main.c $(wildcard kernel1/*.c)
OBJS    := $(SRCS:.c=.o)
HEADERS := $(wildcard kernel1/*.h)

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
