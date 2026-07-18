#ifndef SOLVE_H
#define SOLVE_H

//#define DEBUG_SOLVE

void demo();
int solve(const char *facelets, char *result);
#ifdef DEBUG_SOLVE
void print_table_access(void);
#else
#define print_table_access()
#endif
int binary_table(void);

#endif