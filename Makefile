# ------------------------------------------------------------
# Makefile for Gaussian Blur Benchmark
# ------------------------------------------------------------

CC      := gcc
CFLAGS  := -O3 -march=native -Wall -Wextra
TARGET  := blur_test
SRCS    := $(wildcard *.c)
OBJS    := $(SRCS:.c=.o)
HEADERS := $(wildcard *.h)

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
