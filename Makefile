CXX = gcc
CXXFLAGS= -O0 -g -std=c++11 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas
LDFLAGS=-lm -lstdc++
CXXFILES = src/gcn.cpp src/parser.cpp src/sparse.cpp src/module.cpp src/variable.cpp
HFILES = src/gcn.h src/parser.h src/sparse.h src/module.h src/variable.h

all: seq

seq: src/main.cpp $(CXXFILES) $(HFILES)
	$(CXX) $(CXXFLAGS) -o gcn-seq $(CXXFILES) src/main.cpp $(LDFLAGS)