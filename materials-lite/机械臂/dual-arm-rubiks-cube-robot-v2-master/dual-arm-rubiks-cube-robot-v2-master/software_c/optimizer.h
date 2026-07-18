#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// 每个ACTION最多一次翻面（最长24）+ 一次旋转（最长14）
// (24+14)*25 = 950，预期足够大的空间，最坏情况下也不会溢出
#define MOTION_SEQUENCE_LEN 1024
#define MAX_ACTIONS 25
#define MAX_DP_STATES 96

int solution_to_motion(const char* initial_state, const char* action_str, char* output_sequence);

#endif