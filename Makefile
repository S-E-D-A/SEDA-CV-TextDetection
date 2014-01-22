SHELL = /bin/bash

# Directories
OBJ_DIR		= ./obj
SRC_DIR		= ./src
TEST_DIR	= ./test
GTEST_DIR = ./gtest
LIB_DIR		= ./lib

# Build Objects
SRCS			= $(wildcard $(SRC_DIR)/*.cpp)
INCLUDES	= -I./include -I/usr/include/eigen3 -I./gtest/include -I/usr/include/assimp
OBJS 			= $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Compiler
CXX				= g++
CPPFLAGS	= 
CXXFLAGS	= --std=c++0x -Wextra -Wall $(INCLUDES) $(LIBFLAGS) 
LIBFLAGS  = 

.PHONY: clean all debug

all: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o main

# Targets
$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

debug: CXXFLAGS:=$(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -g 

clean:
	rm -rf *.o
	rm -rf ./obj/*
