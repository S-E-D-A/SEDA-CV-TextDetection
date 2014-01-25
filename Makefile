SHELL = /bin/bash

# Directories
OBJ_DIR		= ./obj
SRC_DIR		= ./src
TEST_DIR	= ./test
GTEST_DIR = ./gtest
LIB_DIR		= ./lib

# Build Objects
SRCS			= $(wildcard $(SRC_DIR)/*.cpp)
INCLUDES	= -I./include -I/usr/include/opencv2
OBJS 			= $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Compiler
CXX				= g++
CPPFLAGS	= 
CXXFLAGS	= --std=c++0x -D_LINUX -msse3 -Wextra -Wall $(INCLUDES) $(LIBFLAGS) 
LIBFLAGS  = -L -/usr/local/lib -llapack -lblas -lopencv_core -lopencv_highgui -lopencv_imgproc 

.PHONY: clean all debug

all: $(OBJS)
	$(CXX) $(OBJS) -o main $(CXXFLAGS) 

# Targets
$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

$(OBJS): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

debug: CXXFLAGS:=$(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -g 
debug: all

clean:
	rm -rf *.o
	rm -rf ./obj/*
