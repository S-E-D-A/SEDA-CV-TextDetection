SHELL = /bin/bash

# Directories
OBJ_DIR		= ./obj
SRC_DIR		= ./src
TEST_DIR	= ./test
GTEST_DIR	= ./gtest
LIB_DIR		= ./lib

# Build Objects
SRCS				= $(wildcard $(SRC_DIR)/*.cpp)
INCLUDES		= -I./include -I/usr/include/opencv2 -I./gtest/include
OBJS				= $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
NONMAINOBJS = $(filter-out ./obj/main.o,$(OBJS))

TESTS			= $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS	= $(TESTS:$(TEST_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Compiler
CXX		    = g++
CPPFLAGS	= -isystem $(GTEST_DIR)/include
CXXFLAGS	= -Wextra -pthread
CXXFLAGS	+= --std=c++0x -D_LINUX -msse3 -Wall -O3 $(INCLUDES) $(LIBFLAGS)
LIBFLAGS	= -L -/usr/local/lib -llapack -lblas -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml

# Google Test
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
								$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS_   = $(GTEST_DIR)/src/*.cc \
								$(GTEST_DIR)/src/*.h \
								$(GTEST_HEADERS)

.PHONY: clean all debug test

all: $(OBJS)
	$(CXX) $(OBJS) -o main $(CXXFLAGS) 

# Google test commands
$(OBJ_DIR)/gtest-all.o : $(GTEST_SRCS_)
		$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
			            $(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o : $(GTEST_SRCS_)
		$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
			            $(GTEST_DIR)/src/gtest_main.cc -o $@

$(OBJ_DIR)/gtest.a : ./obj/gtest-all.o
		$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest_main.a : ./obj/gtest-all.o ./obj/gtest_main.o
		$(AR) $(ARFLAGS) $@ $^

# Targets
$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Google test targets
$(TEST_OBJS): $(OBJ_DIR)/%.o : $(TEST_DIR)/%.cpp $(GTEST_HEADERS)
	$(CXX) $(CXXFLAGS) -g -c $< -o $@ 

test: $(NONMAINOBJS) $(TEST_OBJS) ./obj/gtest_main.a
	$(CXX) $^ $(CXXFLAGS) $(CPPFLAGS) -g -lpthread -o td_test 
	./td_test

debug: CXXFLAGS:=$(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -g 
debug: all


clean:
	rm -rf *.o
	rm -rf ./obj/*
