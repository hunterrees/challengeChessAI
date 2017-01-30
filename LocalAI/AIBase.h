#pragma once

#include <iostream>
#include <sstream>

#include "Move.h"

using namespace std;

class AIBase {
public:
  // Constructors and Destructors
  AIBase(char inBoard[][8], char color) {
    this->color = color;
  }
  ~AIBase() {}

  // Returns the best
  virtual Move getMove() = 0;

  // Prints the board in character form
  string toString() {
    return "";
  }

protected:
  char color;
  //uint16_t board[8][16];
private:
  // Converts the board from and 8*8 array to a long single dimention array
  void convertToEfficientBoard(char inBoard[][8]) {
    for(int i = 0; i < 8; i++) {
      for(int j = 0; j < 8; j++) {

      }
    }
  }

};
