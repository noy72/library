TESTCASE="testcases/testcase-${1}.in"

g++ -std=c++14 main.cpp
./a.out < $TESTCASE
