#ifndef TABLE_H
#define TABLE_H
#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stddef.h>
enum UtilMoves
{
    Ux1 = 0,
    Ux2 = 1,
    Ux3 = 2,
    Rx1 = 3,
    Rx2 = 4,
    Rx3 = 5,
    Fx1 = 6,
    Fx2 = 7,
    Fx3 = 8,
    Dx1 = 9,
    Dx2 = 10,
    Dx3 = 11,
    Lx1 = 12,
    Lx2 = 13,
    Lx3 = 14,
    Bx1 = 15,
    Bx2 = 16,
    Bx3 = 17,
};
    
enum UtilFacelets
{
    U1 = 0,
    U2 = 1,
    U3 = 2,
    U4 = 3,
    U5 = 4,
    U6 = 5,
    U7 = 6,
    U8 = 7,
    U9 = 8,

    R1 = 9,
    R2 = 10,
    R3 = 11,
    R4 = 12,
    R5 = 13,
    R6 = 14,
    R7 = 15,
    R8 = 16,
    R9 = 17,

    F1 = 18,
    F2 = 19,
    F3 = 20,
    F4 = 21,
    F5 = 22,
    F6 = 23,
    F7 = 24,
    F8 = 25,
    F9 = 26,

    D1 = 27,
    D2 = 28,
    D3 = 29,
    D4 = 30,
    D5 = 31,
    D6 = 32,
    D7 = 33,
    D8 = 34,
    D9 = 35,

    L1 = 36,
    L2 = 37,
    L3 = 38,
    L4 = 39,
    L5 = 40,
    L6 = 41,
    L7 = 42,
    L8 = 43,
    L9 = 44,
    
    B1 = 45,
    B2 = 46,
    B3 = 47,
    B4 = 48,
    B5 = 49,
    B6 = 50,
    B7 = 51,
    B8 = 52,
    B9 = 53,
};

enum UtilColors
{
    U = 0,
    R = 1,
    F = 2,
    D = 3,
    L = 4,
    B = 5,
};

// from Search.java
#define USE_TWIST_FLIP_PRUN  true
// Options for research purpose.
#define MAX_PRE_MOVES  20
#define TRY_INVERSE  true
#define TRY_THREE_AXES true
#define USE_COMBP_PRUN USE_TWIST_FLIP_PRUN
#define USE_CONJ_PRUN USE_TWIST_FLIP_PRUN
#define MIN_P1LENGTH_PRE 7
#define MAX_DEPTH2 12
// Verbose_Mask determines if a " . " separates the phase1 and phase2 parts of the solver string like in F' R B R L2 F . U2 U D for example.<br>
#define USE_SEPARATOR 0x1
// Verbose_Mask determines if the solution will be inversed to a scramble/state generator.
#define INVERSE_SOLUTION 0x2
// Verbose_Mask determines if a tag such as "(21f)" will be appended to the solution.
#define APPEND_LENGTH 0x4
// Verbose_Mask determines if guaranteeing the solution to be optimal.
#define OPTIMAL_SOLUTION 0x8

// from CoordCube.java
#define N_MOVES   18
#define N_MOVES2  10

#define N_SLICE  495
#define N_TWIST  2187
#define N_TWIST_SYM  324
#define N_FLIP  2048
#define N_FLIP_SYM  336
#define N_PERM  40320
#define N_PERM_SYM  2768
#define N_MPERM  24
#define N_COMB  (USE_COMBP_PRUN ? 140 : 70)
#define P2_PARITY_MOVE  (USE_COMBP_PRUN ? 0xA5 : 0)

typedef struct structSolution {
    int32_t length;     // 解法总步数
    int32_t depth1;     // 阶段1的步数（用于分隔符显示）
    int32_t verbose;    // 输出控制标志（见宏定义：USE_SEPARATOR等）
    int32_t urfIdx;     // 当前使用的URF变换索引（0-5）
    int32_t moves[31];  // 存储移动序列（每个整数代表一个转动操作）
    bool initOK;        // 初始化标志：true表示解法已计算完成
} Solution;

// 魔方块表示：用角块和棱块描述魔方状态
typedef struct structCubieCube {
    int8_t ca[8];   // 角块状态数组：
                   //  每个元素低3位表示位置（0-7）
                   //  高2位表示方向（0-2）
    int8_t ea[12];  // 棱块状态数组：
                   //  每个元素低4位表示位置（0-11）
                   //  最低位表示方向（0或1）
} CubieCube;

// 坐标表示：用于高效的状态处理和剪枝
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

// 搜索状态结构体：存储整个求解过程的状态
typedef struct structSearch {
    // 移动序列存储
    int32_t move[31];           // 当前搜索路径的移动序列
    
    // 状态缓存（减少重复计算）
    CoordCube nodeUD[21];       // UD轴搜索的状态缓存
    //CoordCube nodeRL[21];       // RL轴搜索的状态缓存（未使用）
    //CoordCube nodeFB[21];       // FB轴搜索的状态缓存（未使用）
    
    // 对称性信息
    int64_t selfSym;            // 魔方自身对称性位图（48位）
    int32_t conjMask;           // 对称变换掩码（跳过无效变换）
    
    // 搜索控制参数
    int32_t urfIdx;             // 当前URF变换索引
    int32_t length1;            // 阶段1计划步数
    int32_t depth1;             // 阶段1当前搜索深度
    int32_t maxDep2;            // 阶段2最大搜索深度
    int32_t solLen;             // 当前找到的最短解法长度
    
    // 解法信息
    Solution solution;          // 找到的解法
    
    // 探针控制（限制搜索复杂度）
    int64_t probe;              // 当前已进行的探针计数
    int64_t probeMax;           // 最大探针限制
    int64_t probeMin;           // 最小探针限制（用于继续搜索）
    
    // 状态控制
    int32_t verbose;            // 输出控制标志
    int32_t valid1;             // 有效缓存深度（避免重复计算）
    bool allowShorter;          // 是否允许更短解法（阶段1）
    
    // 魔方状态
    CubieCube cc;               // 输入的魔方状态（原始）
    CubieCube urfCubieCube[6];  // 6种URF变换后的魔方状态
    CoordCube urfCoordCube[6];  // 6种URF变换后的坐标状态
    
    // 预移动处理
    CubieCube phase1Cubie[21];  // 阶段1各深度的魔方状态缓存
    CubieCube preMoveCubes[MAX_PRE_MOVES + 1]; // 预移动状态缓存
    int32_t preMoves[MAX_PRE_MOVES]; // 预移动序列
    int32_t preMoveLen;          // 当前预移动长度
    int32_t maxPreMoves;         // 最大预移动数
    
    // 搜索模式标志
    bool isRec;                 // 是否在连续搜索模式中
} Search;


typedef struct __attribute__((aligned(8))) ConstantsData {
    // Util 部分
    int32_t __attribute__((aligned(4))) UtilCnk[13][13];
    int32_t __attribute__((aligned(4))) UtilUd2std[18];
    int32_t __attribute__((aligned(4))) UtilStd2ud[18];
    int32_t __attribute__((aligned(4))) UtilCkmv2bit[11];

    // CubieCube 部分
    CubieCube __attribute__((aligned(4))) CubieCubeUrf1;
    CubieCube __attribute__((aligned(4))) CubieCubeUrf2;
    CubieCube __attribute__((aligned(4))) CubieCubeCubeSym[16];
    CubieCube __attribute__((aligned(4))) CubieCubeMoveCube[18];
    int32_t __attribute__((aligned(4)))   CubieCubeFirstMoveSym[48];
    int32_t __attribute__((aligned(4)))   CubieCubeSymMult[16][16];
    int32_t __attribute__((aligned(4)))   CubieCubeSymMultInv[16][16];
    int32_t __attribute__((aligned(4)))   CubieCubeSymMove[16][18];
    int32_t __attribute__((aligned(4)))   CubieCubeSymMoveUD[16][18];
    int64_t __attribute__((aligned(8)))   CubieCubeMoveCubeSym[18];
    int32_t __attribute__((aligned(4)))   CubieCubeSym8Move[8 * 18];
    int8_t __attribute__((aligned(4)))    CubieCubeUrfMove[6][18];
    uint16_t __attribute__((aligned(4)))  CubieCubePermInvEdgeSym[2768];
    uint16_t __attribute__((aligned(4)))  CubieCubeFlipR2S[2048];
    uint16_t __attribute__((aligned(4)))  CubieCubeEPermR2S[40320];
    int8_t __attribute__((aligned(4)))    CubieCubePerm2CombP[2768];
    int8_t __attribute__((aligned(4)))    CubieCubeMPermInv[24];
    uint16_t __attribute__((aligned(4)))  CubieCubeFlipS2RF[2688];
    uint16_t __attribute__((aligned(4)))  CubieCubeTwistR2S[2187];

    // CoordCube 部分
    uint16_t __attribute__((aligned(4))) CoordCubeUDSliceMove[495][18];
    uint16_t __attribute__((aligned(4))) CoordCubeTwistMove[324][18];
    uint16_t __attribute__((aligned(4))) CoordCubeFlipMove[336][18];
    uint16_t __attribute__((aligned(4))) CoordCubeUDSliceConj[495][8];
    int32_t __attribute__((aligned(4)))  CoordCubeUDSliceTwistPrun[20048];
    int32_t __attribute__((aligned(4)))  CoordCubeUDSliceFlipPrun[20791];
    uint16_t __attribute__((aligned(4))) CoordCubeCPermMove[2768][10];
    uint16_t __attribute__((aligned(4))) CoordCubeEPermMove[2768][10];
    uint16_t __attribute__((aligned(4))) CoordCubeMPermMove[24][10];
    uint16_t __attribute__((aligned(4))) CoordCubeMPermConj[24][16];
    uint16_t __attribute__((aligned(4))) CoordCubeCCombPConj[140][16];
    int32_t __attribute__((aligned(4)))  CoordCubeMCPermPrun[8305];
    int32_t __attribute__((aligned(4)))  CoordCubeEPermCCombPPrun[48441];
    int32_t __attribute__((aligned(4)))  CoordCubeTwistFlipPrun[82945];
} ConstantsData;

extern const CubieCube CubieCubeUrf1;
extern const CubieCube CubieCubeUrf2;

// 16 symmetries generated by S_F2, S_U4 and S_LR2
extern const CubieCube CubieCubeCubeSym [16];
// 18 move cubes
extern const CubieCube CubieCubeMoveCube[18];

// 组合数表 C(n,k) 用于计算组合数
extern const int32_t UtilCnk[13][13];

// UD转动与标准转动的映射表（只使用U/D面和双层转动）
extern const int32_t UtilUd2std[18]; // UD转动->标准转动
extern const int32_t UtilStd2ud[18]; // 标准转动->UD转动

// 移动有效性检查表（避免连续同轴转动）
extern const int32_t UtilCkmv2bit[11];

// 对称性相关表
extern const int64_t CubieCubeMoveCubeSym[18];  // 每个基本转动的对称性标记
extern const int32_t CubieCubeFirstMoveSym[48]; // 对称性对应的第一步限制，不确定这个有没有用，测试10000组随机打乱的魔方，都没有访问到这个表
extern const int32_t CubieCubeSymMult[16][16];  // 对称变换乘法表
extern const int32_t CubieCubeSymMultInv[16][16]; // 对称变换逆乘法表
extern const int32_t CubieCubeSymMove[16][18];   // 对称变换后的转动映射
extern const int32_t CubieCubeSym8Move[8 * 18];  // 8种对称下的转动映射（压缩形式）
extern const int32_t CubieCubeSymMoveUD[16][18]; // UD转动在对称变换后的映射

// URF变换转动映射（6种基准面变换）
extern const int8_t  CubieCubeUrfMove[6][18];

// 坐标转换表
extern const uint16_t CubieCubePermInvEdgeSym[2768]; // 棱块排列对称逆变换
extern const uint16_t CubieCubeFlipR2S[2048];       // Flip原始坐标->对称坐标
extern const uint16_t CubieCubeTwistR2S[2187];      // Twist原始坐标->对称坐标
extern const uint16_t CubieCubeEPermR2S[40320];     // 棱块排列原始坐标->对称坐标
extern const int8_t  CubieCubePerm2CombP[2768];     // 角块排列->组合坐标
extern const int8_t  CubieCubeMPermInv[24];         // 中棱块排列逆变换

// 移动表（状态转移）
extern const uint16_t CoordCubeUDSliceMove[495][18];  // UDSlice坐标移动表
extern const uint16_t CoordCubeTwistMove[324][18];    // Twist坐标移动表
extern const uint16_t CoordCubeFlipMove[336][18];     // Flip坐标移动表
extern const uint16_t CoordCubeUDSliceConj[495][8];   // UDSlice对称共轭
extern const int32_t CoordCubeUDSliceTwistPrun[20048];   // 角块排列移动表（UD转动）
extern const int32_t CoordCubeUDSliceFlipPrun[20791];   // 棱块排列移动表（UD转动）
extern const uint16_t CoordCubeCPermMove[2768][10];     // 中棱块排列移动表（UD转动）
extern const uint16_t CoordCubeEPermMove[2768][10];     // 中棱块对称共轭
extern const uint16_t CoordCubeMPermMove[24][10];   // 角块组合对称共轭

// 剪枝表（加速搜索）
extern const uint16_t CoordCubeMPermConj[24][16];    // UDSlice+Twist剪枝
extern const uint16_t CoordCubeCCombPConj[140][16];     // UDSlice+Flip剪枝
extern const int32_t CoordCubeMCPermPrun[8305];           // 中棱+角块排列剪枝
extern const int32_t CoordCubeEPermCCombPPrun[48441];     // 棱块排列+角块组合剪枝
extern const uint16_t CubieCubeFlipS2RF[2688];            // Flip对称坐标转换
extern const int32_t CoordCubeTwistFlipPrun[82945];       // Twist+Flip组合剪枝

#endif