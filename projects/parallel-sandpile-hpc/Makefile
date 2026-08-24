# Makefile for parallel-sandpile-hpc
#
# On a normal machine: make
# On ARCHER2: module load PrgEnv-gnu (or cray) first — cc wraps mpicc there.

CC      = mpicc
CFLAGS  = -O2 -Wall -std=c99
SRC_DIR = src
OBJS    = $(SRC_DIR)/sandpile_2d.o $(SRC_DIR)/arraymalloc.o $(SRC_DIR)/sandio.o
TARGET  = sandpile_2d

.PHONY: all clean run-local test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Quick local smoke test with 4 oversubscribed ranks (for a laptop, not ARCHER2)
run-local: $(TARGET)
	mpirun --oversubscribe -np 4 ./$(TARGET) --size 64 --maxh 4 --steps 5000 --out sand.ppm

test: $(TARGET)
	python3 -m pytest tests/ -v

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET) *.ppm
