#pragma once

#define MAX_PRE_MOVES    20  // 最大预移动步数
#define MIN_P1LENGTH_PRE 7   // 阶段1的最小长度（含预移动）
#define MAX_DEPTH2       12  // 阶段2的最大搜索深度
#define N_SLICE          495 // UD切片坐标的数量
#define N_MPERM          24  // 中层棱块排列的数量
#define N_COMB           140 // 组合坐标的数量

// 解法结构体：存储求解结果
typedef struct structSolution {
    int32_t length;     // 解法总步数
    int32_t urfIdx;     // 当前使用的URF变换索引（0-5）
    int32_t moves[31];  // 存储移动序列（每个整数代表一个转动操作）
    bool initOK;        // 初始化标志：true表示解法已计算完成
} Solution;

// 坐标表示：用于高效的状态处理和剪枝（用于一阶段）
typedef struct structCoordCube {
    int32_t twist;   // 角块方向坐标（原始或对称压缩）
    int32_t tsym;    // 角块对称性标记（低3位）
    int32_t flip;    // 棱块方向坐标（原始或对称压缩）
    int32_t fsym;    // 棱块对称性标记（低3位）
    int32_t slice;   // UD中层棱块位置坐标（0-494）
    int32_t prun;    // 剪枝值（到目标状态的最小估计步数）

    // 用于对称剪枝的额外坐标
    int32_t twistc;  // 共轭变换后的角块方向坐标
    int32_t flipc;   // 共轭变换后的棱块方向坐标
} CoordCube;

// 魔方块表示：用角块和棱块描述魔方状态
typedef struct structCubieCube {
    int8_t ep[12];// 棱块位置（0-11）
    int8_t er[12];// 棱块方向（0-1）
    int8_t cp[8]; // 角块位置（0-7）
    int8_t cr[8]; // 角块方向（0-2）
} CubieCube;

// 搜索状态结构体：存储整个求解过程的状态
typedef struct structSearch {
    // 移动序列存储
    int32_t move[31];           // 当前搜索路径的移动序列
    
    // 状态缓存（减少重复计算）
    CoordCube nodeUD[21];       // UD轴搜索的状态缓存（用于一阶段）
    
    // 搜索控制参数
    int32_t urfIdx;             // 当前URF变换索引
    int32_t length1;            // 阶段1计划步数
    int32_t depth1;             // 阶段1当前搜索深度
    int32_t maxDep2;            // 阶段2最大搜索深度
    int32_t solLen;             // 当前找到的最短解法长度
    
    // 解法信息
    Solution solution;          // 找到的解法
    
    // 探针控制（限制搜索复杂度，用于二阶段）
    int probe;                  // 调用SearchInitPhase2Pre的次数
    int probeMax;               // 最大探针限制
    int probeMin;               // 最小探针限制（用于继续搜索）
    
    // 状态控制
    int32_t valid1;             // 有效缓存深度（避免重复计算）
    bool allowShorter;          // 是否允许更短解法（阶段1）
    
    // 魔方状态
    CubieCube cc;               // 输入的魔方状态（原始）
    CubieCube urfCubieCube[6];  // 6种URF变换后的魔方状态（用于一阶段）
    CoordCube urfCoordCube[6];  // 6种URF变换后的坐标状态（用于一阶段）
    
    // 预移动处理
    CubieCube phase1Cubie[21];  // 阶段1各深度的魔方状态缓存
    CubieCube preMoveCubes[MAX_PRE_MOVES + 1]; // 预移动状态缓存
    int32_t preMoves[MAX_PRE_MOVES]; // 预移动序列
    int32_t preMoveLen;          // 当前预移动长度
    int32_t maxPreMoves;         // 最大预移动数

    bool isRec;
} Search;

int SearchNext(Search *this, char *solution_buffer, int probeMax, int probeMin);
int SearchSolution(Search *this, char *solution_buffer, const char *facelets, int maxDepth, int probeMax, int probeMin);
