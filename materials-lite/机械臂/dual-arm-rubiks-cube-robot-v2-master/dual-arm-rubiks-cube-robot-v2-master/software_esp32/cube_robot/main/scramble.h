#pragma once
#define SOLVED_CUBE "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
int verify(const char *from, const char *to, const char *solution);
void scramble(char *cube_str);
int reverse_solution(const char *solution_in, char *solution_out);
