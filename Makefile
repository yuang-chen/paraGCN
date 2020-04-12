CXX = gcc
CXXFLAGS= -O0 -g -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++

CXXFILES = src/gcn.cpp src/parser.cpp src/sparse.cpp src/module.cpp src/variable.cpp src/rand.cpp src/optim.cpp src/timer.cpp
HFILES = src/gcn.h src/parser.h src/sparse.h src/module.h src/variable.h src/rand.h src/optim.h src/timer.h

TEST_CXXFILES=test/module_test.cpp test/util.cpp test/optim_test.cpp
TEST_HFILES=test/util.h

all: seq

seq: src/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)

test: $(CXXFILES) $(HFILES) $(TEST_CXXFILES) $(TEST_HFILES)
	$(CXX) $(CXXFLAGS) -Iinclude -o gcn-test $(CXXFILES) $(TEST_CXXFILES) test/main.cpp $(LDFLAGS)
