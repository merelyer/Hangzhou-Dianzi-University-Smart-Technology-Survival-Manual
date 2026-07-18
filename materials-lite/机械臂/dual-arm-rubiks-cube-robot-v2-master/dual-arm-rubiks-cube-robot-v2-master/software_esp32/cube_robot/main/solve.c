// 基于min2phase https://github.com/cs0x7f/min2phase 修改
// 没有移植生成table的逻辑，如果需要请参考原始项目
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "solve.h"
#include "table.h"

#include "esp_log.h"
static const char *TAG = "solve";

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// 将解法转换为字符串表示
static char *SolutionToString(Solution *this, char *buffer) {
    char *sb = buffer;
    const char move2str[18][3] = {
        "U ", "U2", "U'", "R ", "R2", "R'", "F ", "F2", "F'",
        "D ", "D2", "D'", "L ", "L2", "L'", "B ", "B2", "B'"
    };
    int urf = this->urfIdx;
    if (urf < 3) {
        // 正向应用URF变换
        for (int s = 0; s < this->length; s++) {
            strcpy(sb, move2str[CubieCubeUrfMove[urf][this->moves[s]]]);
            sb += 2;
            strcpy(sb, " ");
            sb += 1;
        }
    } else {
        // 逆向应用URF变换
        for (int s = this->length - 1; s >= 0; s--) {
            strcpy(sb, move2str[CubieCubeUrfMove[urf][this->moves[s]]]);
            sb += 2;
            strcpy(sb, " ");
            sb += 1;
        }
    }
    return buffer;
}

// 设置解法参数：URF索引和阶段1深度
static void SolutionSetArgs(Solution *this, int urfIdx, int depth1) {
    this->urfIdx = urfIdx;
}

// 向解法中添加移动步（自动合并相邻同面操作）
static void SolutionAppendSolMove(Solution *this, int curMove) {
    if (this->length == 0) {
        this->moves[this->length++] = curMove;
        return;
    }
    int axisCur = curMove / 3; // 当前移动的面（U=0, R=1, F=2等）
    int axisLast = this->moves[this->length - 1] / 3; // 上一步移动的面
    
    // 如果当前移动与上一步在同一面，则合并
    // 覆盖率测试测不到合并后抵消的分支，不确定是否会出现这种情况
    if (axisCur == axisLast) {
        int pow = (curMove % 3 + this->moves[this->length - 1] % 3 + 1) % 4;
        if (pow == 3) { // 合并后抵消
            this->length--;
        } else { // 更新上一步的转动角度
            this->moves[this->length - 1] = axisCur * 3 + pow;
        }
        return;
    }
    
    // 检查三步合并（如R R' R变为R）
    if (this->length > 1
            && axisCur % 3 == axisLast % 3
            && axisCur == this->moves[this->length - 2] / 3) {
        int pow = (curMove % 3 + this->moves[this->length - 2] % 3 + 1) % 4;
        if (pow == 3) { // 合并后抵消
            this->moves[this->length - 2] = this->moves[this->length - 1];
            this->length--;
        } else { // 更新三步中的第一步
            this->moves[this->length - 2] = axisCur * 3 + pow;
        }
        return;
    }
    
    // 无法合并，直接添加新步
    this->moves[this->length++] = curMove;
}


/**
 * 注意：棱块排列坐标和角块排列坐标具有相同的对称结构。
 * 因此它们的ClassIndexToRepresentantArray是相同的。
 * 当x是原始棱块排列坐标时，y*16+k是棱块对称坐标，y*16+(k^e2c[k])将
 * 是相同状态下（原始角块排列坐标为x）的角块对称坐标。
 */
static int CubieCubeESym2CSym(int idx) {
    const int SYM_E2C_MAGIC = 0x00DDDD00;
    return idx ^ (SYM_E2C_MAGIC >> ((idx & 0xf) << 1) & 3);
}

// ********************************************* 坐标获取和设置函数 *********************************************
// XSym：X的对称坐标。必须在ClassIndexToRepresentantArrays初始化后调用。

// ++++++++++++++++++++ 阶段1坐标 ++++++++++++++++++++
// Flip：12个棱块的方向。原始[0, 2048) 对称[0, 336*8)
// Twist：8个角块的方向。原始[0, 2187) 对称[0, 324*8)
// UDSlice：4个UD中层棱块的位置（忽略顺序）。[0, 495)

// 获取棱块方向坐标（原始）
static int CubieCubeGetFlip(CubieCube *this) {
    int idx = 0;
    for (int i = 0; i < 11; i++) {
        idx = idx << 1 | (this->er[i]);
    }
    return idx;
}

// 获取棱块方向对称坐标
static int CubieCubeGetFlipSym(CubieCube *this) {
    return CubieCubeFlipR2S[CubieCubeGetFlip(this)];
}

// 获取角块方向坐标（原始）
static int CubieCubeGetTwist(CubieCube *this) {
    int idx = 0;
    for (int i = 0; i < 7; i++) {
        idx += (idx << 1) + (this->cr[i]);
    }
    return idx;
}

// 获取角块方向对称坐标
static int CubieCubeGetTwistSym(CubieCube *this) {
    return CubieCubeTwistR2S[CubieCubeGetTwist(this)];
}

// 计算组合坐标（用于UD切片）
static int UtilGetComb(int8_t arr[], int arr_length, int mask) {
    int end = arr_length - 1;
    int idxC = 0, r = 4; // 选择4个UD中层棱块
    for (int i = end; i >= 0; i--) {
        int perm = arr[i];
        if ((perm & 0xc) == mask) { // 检查是否为UD中层棱块
            idxC += UtilCnk[i][r--]; // 累加组合数
        }
    }
    return idxC;
}

// 获取UD切片坐标
static int CubieCubeGetUDSlice(CubieCube *this) {
    return 494 - UtilGetComb(this->ep, 12, 8);
}

// ++++++++++++++++++++ 阶段2坐标 ++++++++++++++++++++
// EPerm：8个UD棱块的排列。原始[0,40320) 对称[0,2187*16)
// Cperm：8个角块的排列。原始[0,40320) 对称[0,2187*16)
// MPerm：4个UD中层棱块的排列。[0,24)

// 计算排列索引（n个元素的排列）
static int UtilGetNPerm(int8_t arr[], int n) 
{
    int idx = 0;
    int64_t val = 0xFEDCBA9876543210L; // 用于高效排列计算的位图
    for (int i = 0; i < n - 1; i++) {
        int v = arr[i] << 2; // 每个元素占4位
        idx = (n - i) * idx + (int) (val >> v & 0xf); // 累加位置值
        val -= 0x1111111111111110L << v; // 移除已用元素
    }
    return idx;
}

// 获取角块排列坐标（原始）
static int CubieCubeGetCPerm(CubieCube *this) {
    return UtilGetNPerm(this->cp, 8);
}

// 获取棱块排列坐标（原始）
static int CubieCubeGetEPerm(CubieCube *this) {
    return UtilGetNPerm(this->ep, 8);
}

// 获取中层棱块排列坐标
static int CubieCubeGetMPerm(CubieCube *this) {
    return UtilGetNPerm(this->ep, 12) % 24;
}

// 获取角块排列对称坐标
static int CubieCubeGetCPermSym(CubieCube *this) {
    return CubieCubeESym2CSym(CubieCubeEPermR2S[CubieCubeGetCPerm(this)]);
}

// 获取棱块排列对称坐标
static int CubieCubeGetEPermSym(CubieCube *this) {
    return CubieCubeEPermR2S[CubieCubeGetEPerm(this)];
}


/**
 * prod = a * b, 仅角块
 */
static void CubieCubeCornMult(const CubieCube *a, const CubieCube *b, CubieCube *prod) {
    for (int corn = 0; corn < 8; corn++) {
        int oriA = a->cr[b->cp[corn]];       // b中corn位置角块在a中的方向
        int oriB = b->cr[corn];              // b中corn位置角块的原始方向
        prod->cp[corn] = a->cp[b->cp[corn]]; // 位置复合
        prod->cr[corn] = (oriA + oriB) % 3;  // 方向复合（模3）
    }
}

/**
 * prod = a * b, 仅棱块
 */
static void CubieCubeEdgeMult(const CubieCube *a, const CubieCube *b, CubieCube *prod) {
    for (int ed = 0; ed < 12; ed++) {
        int8_t index = b->ep[ed]; // b中ed位置棱块的目标位置
        prod->ep[ed] = a->ep[index]; // a中该位置的棱块最终位置
        prod->er[ed] = a->er[index] ^ (b->er[ed]); // 方向异或（0或1）
    }
}

/**
 * this = S_urf^-1 * this * S_urf（URF共轭变换）
 */
static void CubieCubeURFConjugate(CubieCube *this) {
    CubieCube temps={0};
    CubieCubeCornMult(&CubieCubeUrf2, this, &temps);  // 前乘S_urf^-1
    CubieCubeCornMult(&temps, &CubieCubeUrf1, this);  // 后乘S_urf
    CubieCubeEdgeMult(&CubieCubeUrf2, this, &temps);  // 前乘S_urf^-1
    CubieCubeEdgeMult(&temps, &CubieCubeUrf1, this);  // 后乘S_urf
}

// 计算魔方状态的逆
static void CubieCubeInvCubieCube(CubieCube *this) {
    CubieCube temps={0};
    // 棱块：位置和方向取反
    for (int8_t edge = 0; edge < 12; edge++) {
        int8_t index = this->ep[edge]; // 当前位置的棱块原始位置
        temps.er[index] = this->er[edge]; // 方向直接复制
        temps.ep[index] = edge;           // 位置取反
    }
    // 角块：位置取反，方向调整（0->0, 1->2, 2->1）
    for (int8_t corn = 0; corn < 8; corn++) {
        temps.cp[this->cp[corn]] = corn; // 位置取反
        temps.cr[this->cp[corn]] = (0x4 >> this->cr[corn]) & 0x3; // 方向调整
    }
    memcpy(this, &temps, sizeof(CubieCube));
}

/**
 * b = S_idx^-1 * a * S_idx, 仅角块
    */
static void CubieCubeCornConjugate(CubieCube *a, int idx, CubieCube *b) {
    const CubieCube *sinv = &CubieCubeCubeSym[CubieCubeSymMultInv[0][idx]]; // 对称操作逆
    const CubieCube *s = &CubieCubeCubeSym[idx]; // 对称操作
    for (int corn = 0; corn < 8; corn++) {
        int oriA = sinv->cr[a->cp[s->cp[corn]]]; // 复合后的方向
        int oriB = a->cr[s->cp[corn]];           // 原始方向
        int ori = (oriA < 3) ? oriB : (3 - oriB) % 3; // 方向复合计算
        b->cp[corn] = sinv->cp[a->cp[s->cp[corn]]]; // 位置复合
        b->cr[corn] = ori; // 设置最终方向
    }
}

/**
 * b = S_idx^-1 * a * S_idx, 仅棱块
    */
static void CubieCubeEdgeConjugate(CubieCube *a, int idx, CubieCube *b) {
    const CubieCube *sinv = &CubieCubeCubeSym[CubieCubeSymMultInv[0][idx]]; // 对称操作逆
    const CubieCube *s = &CubieCubeCubeSym[idx]; // 对称操作
    for (int ed = 0; ed < 12; ed++) {
        b->ep[ed] = sinv->ep[a->ep[s->ep[ed]]]; // 位置复合
        b->er[ed] = sinv->er[a->ep[s->ep[ed]]] ^ (a->er[s->ep[ed]]) ^ (s->er[ed]); // 方向复合（异或）
    }
}

// 获取对称排列的逆
static int CubieCubeGetPermSymInv(int idx, int sym, bool isCorner) {
    int idxi = CubieCubePermInvEdgeSym[idx]; // 原始排列的逆
    if (isCorner) {
        idxi = CubieCubeESym2CSym(idxi); // 如果是角块，转换坐标
    }
    return (idxi & 0xfff0) | CubieCubeSymMult[idxi & 0xf][sym]; // 组合对称索引
}

// -------------------- 坐标魔方操作 ------------------------ //

// 从剪枝表中获取剪枝值
static int CoordCubeGetPruning(const int32_t table[], int index) {
    // 计算表索引和位移
    return table[index >> 3] >> ((index & 7) << 2) & 0xf; 
}

/**
 * 执行移动并更新剪枝值（阶段1）
 * @return 更新后的剪枝值
 */
static int CoordCubeDoMovePrun(CoordCube *this, CoordCube *cc, int m, bool isPhase1) {
    // 更新UD切片坐标
    this->slice = CoordCubeUDSliceMove[cc->slice][m];
    
    // 更新棱块方向坐标（考虑对称性）
    this->flip = CoordCubeFlipMove[cc->flip][CubieCubeSym8Move[m << 3 | cc->fsym]];
    this->fsym = (this->flip & 7) ^ cc->fsym; // 更新对称标记
    this->flip >>= 3;
    
    // 更新角块方向坐标（考虑对称性）
    this->twist = CoordCubeTwistMove[cc->twist][CubieCubeSym8Move[m << 3 | cc->tsym]];
    this->tsym = (this->twist & 7) ^ cc->tsym; // 更新对称标记
    this->twist >>= 3;
    
    // 计算新状态的剪枝值（取三种坐标的最大剪枝值）
    this->prun = max(
                max(
                    CoordCubeGetPruning(CoordCubeUDSliceTwistPrun,
                                this->twist * N_SLICE + CoordCubeUDSliceConj[this->slice][this->tsym]),
                    CoordCubeGetPruning(CoordCubeUDSliceFlipPrun,
                                this->flip * N_SLICE + CoordCubeUDSliceConj[this->slice][this->fsym])),
                    CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                                this->twist << 11 | CubieCubeFlipS2RF[this->flip << 3 | (this->fsym ^ this->tsym)]));
    return this->prun;
}

// 执行移动并更新共轭坐标的剪枝值（对称剪枝）
static int CoordCubeDoMovePrunConj(CoordCube *this, CoordCube *cc, int m) {
    m = CubieCubeSymMove[3][m]; // 应用对称变换
    // 更新棱块方向共轭坐标
    this->flipc = CoordCubeFlipMove[cc->flipc >> 3][CubieCubeSym8Move[m << 3 | (cc->flipc & 7)]] ^ (cc->flipc & 7);
    // 更新角块方向共轭坐标
    this->twistc = CoordCubeTwistMove[cc->twistc >> 3][CubieCubeSym8Move[m << 3 | (cc->twistc & 7)]] ^ (cc->twistc & 7);
    // 计算新状态的剪枝值
    return CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                        (this->twistc >> 3) << 11 | CubieCubeFlipS2RF[this->flipc ^ (this->twistc & 7)]);
}

// 设置坐标状态并计算初始剪枝值
static bool CoordCubeSetWithPrun(CoordCube *this, CubieCube *cc, int depth) {
    // 获取角块方向对称坐标
    this->twist = CubieCubeGetTwistSym(cc);
    // 获取棱块方向对称坐标
    this->flip = CubieCubeGetFlipSym(cc);
    this->tsym = this->twist & 7;// 提取对称标记
    this->twist = this->twist >> 3;// 提取坐标值
    // 计算初始剪枝值（基于角块和棱块方向）
    this->prun = CoordCubeGetPruning(CoordCubeTwistFlipPrun, this->twist << 11 | CubieCubeFlipS2RF[this->flip ^ this->tsym]);
    if (this->prun > depth) {
        return false; // 剪枝：超过深度限制
    }

    // 获取UD切片坐标
    this->fsym = this->flip & 7;
    this->flip = this->flip >> 3;
    this->slice = CubieCubeGetUDSlice(cc);
    
    // 更新剪枝值（加入UD切片坐标）
    this->prun = max(this->prun, max(
                        CoordCubeGetPruning(CoordCubeUDSliceTwistPrun,
                                    this->twist * N_SLICE + CoordCubeUDSliceConj[this->slice][this->tsym]),
                        CoordCubeGetPruning(CoordCubeUDSliceFlipPrun,
                                    this->flip * N_SLICE + CoordCubeUDSliceConj[this->slice][this->fsym])));
    if (this->prun > depth) {
        return false; // 剪枝：超过深度限制
    }

    // 计算共轭坐标（用于对称剪枝）
    CubieCube pc = {};
    CubieCubeCornConjugate(cc, 1, &pc); // 应用对称操作1
    CubieCubeEdgeConjugate(cc, 1, &pc); // 应用对称操作1
    this->twistc = CubieCubeGetTwistSym(&pc); // 获取角块共轭坐标
    this->flipc = CubieCubeGetFlipSym(&pc);   // 获取棱块共轭坐标
    
    // 更新剪枝值（加入共轭坐标）
    this->prun = max(this->prun,
                    CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                                (this->twistc >> 3) << 11 | CubieCubeFlipS2RF[this->flipc ^ (this->twistc & 7)]));
    return this->prun <= depth; // 返回是否在深度限制内
}

// 阶段2搜索函数
// 返回：-1 未找到解法；其他：解法长度调整值
static int SearchPhase2(int32_t move[31], int edge, int esym, int corn, int csym, int mid, int maxl, int depth, int lm) {
    // 检查是否达到目标状态（所有坐标均为0）
    if (edge == 0 && corn == 0 && mid == 0) {
        return maxl; // 返回剩余步数（表示解法比预期短）
    }
    
    int moveMask = UtilCkmv2bit[lm]; // 获取禁止移动位图
    for (int m = 0; m < 10; m++) { // 遍历10种UD转动（阶段2只允许U,D,R2,F2,L2,B2）
        if ((moveMask >> m & 1) != 0) { // 跳过禁止移动
            m += 0x42 >> m & 3; // 跳过同面移动
            continue;
        }
        
        // 执行移动并更新坐标
        int midx = CoordCubeMPermMove[mid][m];
        int cornx = CoordCubeCPermMove[corn][CubieCubeSymMoveUD[csym][m]];
        int csymx = CubieCubeSymMult[cornx & 0xf][csym];
        cornx >>= 4;
        int edgex = CoordCubeEPermMove[edge][CubieCubeSymMoveUD[esym][m]];
        int esymx = CubieCubeSymMult[edgex & 0xf][esym];
        edgex >>= 4;
        
        // 计算对称逆坐标（用于剪枝）
        int edgei = CubieCubeGetPermSymInv(edgex, esymx, false);
        int corni = CubieCubeGetPermSymInv(cornx, csymx, true);
        
        // 剪枝检查
        int prun = CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        (edgei >> 4) * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[corni >> 4] & 0xff][CubieCubeSymMultInv[edgei & 0xf][corni & 0xf]]);
        if (prun > maxl + 1) {
            return maxl - prun + 1; // 剪枝：超出最大步数
        } else if (prun >= maxl) {
            m += 0x42 >> m & 3 & (maxl - prun); // 跳过同面移动
            continue;
        }
        
        // 二次剪枝检查
        prun = max(
                    CoordCubeGetPruning(CoordCubeMCPermPrun,
                                        cornx * N_MPERM + CoordCubeMPermConj[midx][csymx]),
                    CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        edgex * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[cornx] & 0xff][CubieCubeSymMultInv[esymx][csymx]]));
        if (prun >= maxl) {
            m += 0x42 >> m & 3 & (maxl - prun); // 跳过同面移动
            continue;
        }
        
        // 递归搜索
        int ret = SearchPhase2(move, edgex, esymx, cornx, csymx, midx, maxl - 1, depth + 1, m);
        if (ret >= 0) { // 找到解法
            move[depth] = UtilUd2std[m]; // 记录移动
            return ret; // 返回调整值
        }
        if (ret < -2) { // 需要提前退出
            break;
        }
        if (ret < -1) { // 跳过同面移动
            m += 0x42 >> m & 3;
        }
    }
    return -1; // 未找到解法
}

// 初始化阶段2搜索
static  int SearchInitPhase2(Search *this, int p2corn, int p2csym, int p2edge, int p2esym, int p2mid, int edgei, int corni) {
    // 计算初始剪枝值
    int prun = max(
                    CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        (edgei >> 4) * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[corni >> 4] & 0xff][CubieCubeSymMultInv[edgei & 0xf][corni & 0xf]]),
                    max(
                        CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                            p2edge * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[p2corn] & 0xff][CubieCubeSymMultInv[p2esym][p2csym]]),
                        CoordCubeGetPruning(CoordCubeMCPermPrun,
                                            p2corn * N_MPERM + CoordCubeMPermConj[p2mid][p2csym])));

    if (prun > this->maxDep2) { // 剪枝：超过阶段2最大深度
        return prun - this->maxDep2;
    }

    int depth2;
    // 从最大深度向下搜索
    for (depth2 = this->maxDep2; depth2 >= prun; depth2--) {
        // 调用阶段2搜索
        int ret = SearchPhase2(this->move, p2edge, p2esym, p2corn, p2csym, p2mid, depth2, this->depth1, 10);
        if (ret < 0) { // 未找到解法
            break;
        }
        depth2 -= ret; // 调整解法长度
        // 记录解法
        this->solLen = 0;
        memset(&this->solution, 0, sizeof(Solution));
        this->solution.initOK = true;
        SolutionSetArgs(&this->solution, this->urfIdx, this->depth1);
        // 添加阶段1和阶段2的移动
        for (int i = 0; i < this->depth1 + depth2; i++) {
            SolutionAppendSolMove(&this->solution, this->move[i]);
        }
        // 添加预移动（逆序）
        for (int i = this->preMoveLen - 1; i >= 0; i--) {
            SolutionAppendSolMove(&this->solution, this->preMoves[i]);
        }
        this->solLen = this->solution.length; // 更新解法长度
    }

    if (depth2 != this->maxDep2) { // 找到解法
        this->maxDep2 = min(MAX_DEPTH2, this->solLen - this->length1 - 1); // 更新阶段2最大深度
        return this->probe >= this->probeMin ? 0 : 1; // 检查探针限制
    }
    return 1; // 未找到解法
}

/**
 * 初始化阶段2搜索（考虑预移动调整）
 * @return
 *      0: 找到解法或超过探针限制
 *      1: 继续搜索
 */
static int SearchInitPhase2Pre(Search *this) {
    this->isRec = false;
    if (this->probe >= (this->solution.initOK == false ? this->probeMax : this->probeMin)) {
        return 0; // 超过探针限制
    }
    ++this->probe; // 增加探针计数
    // printf("SearchInitPhase2Pre probe = %d\n", this->probe);

    // 更新阶段1状态缓存
    for (int i = this->valid1; i < this->depth1; i++) {
        CubieCubeCornMult(&this->phase1Cubie[i], &CubieCubeMoveCube[this->move[i]], &this->phase1Cubie[i + 1]);
        CubieCubeEdgeMult(&this->phase1Cubie[i], &CubieCubeMoveCube[this->move[i]], &this->phase1Cubie[i + 1]);
    }
    this->valid1 = this->depth1; // 更新有效深度

    // 获取阶段1结束时的坐标
    int p2corn = CubieCubeGetCPermSym(&this->phase1Cubie[this->depth1]);
    int p2csym = p2corn & 0xf;
    p2corn >>= 4;
    int p2edge = CubieCubeGetEPermSym(&this->phase1Cubie[this->depth1]);
    int p2esym = p2edge & 0xf;
    p2edge >>= 4;
    int p2mid = CubieCubeGetMPerm(&this->phase1Cubie[this->depth1]);
    int edgei = CubieCubeGetPermSymInv(p2edge, p2esym, false);
    int corni = CubieCubeGetPermSymInv(p2corn, p2csym, true);

    int lastMove = this->depth1 == 0 ? -1 : this->move[this->depth1 - 1];
    int lastPre = this->preMoveLen == 0 ? -1 : this->preMoves[this->preMoveLen - 1];

    int ret = 0;
    int p2switchMax = (this->preMoveLen == 0 ? 1 : 2) * (this->depth1 == 0 ? 1 : 2);
    // 尝试不同的调整方案（正常/调整最后一步/调整预移动）
    for (int p2switch = 0, p2switchMask = (1 << p2switchMax) - 1;
            p2switch < p2switchMax; p2switch++) {
        // 0:正常; 1:调整最后一步; 2:调整最后一步+预移动; 3:调整预移动
        if ((p2switchMask >> p2switch & 1) != 0) {
            p2switchMask &= ~(1 << p2switch);
            ret = SearchInitPhase2(this, p2corn, p2csym, p2edge, p2esym, p2mid, edgei, corni);
            if (ret == 0 || ret > 2) {
                break;
            } else if (ret == 2) {
                p2switchMask &= 0x4 << p2switch; // 跳过无效调整
            }
        }
        if (p2switchMask == 0) {
            break;
        }
        // 调整最后一步移动
        if ((p2switch & 1) == 0 && this->depth1 > 0) {
            int m = UtilStd2ud[lastMove / 3 * 3 + 1]; // 转换为UD轴移动
            this->move[this->depth1 - 1] = UtilUd2std[m] * 2 - this->move[this->depth1 - 1]; // 调整转动角度
            // 更新坐标
            p2mid = CoordCubeMPermMove[p2mid][m];
            p2corn = CoordCubeCPermMove[p2corn][CubieCubeSymMoveUD[p2csym][m]];
            p2csym = CubieCubeSymMult[p2corn & 0xf][p2csym];
            p2corn >>= 4;
            p2edge = CoordCubeEPermMove[p2edge][CubieCubeSymMoveUD[p2esym][m]];
            p2esym = CubieCubeSymMult[p2edge & 0xf][p2esym];
            p2edge >>= 4;
            corni = CubieCubeGetPermSymInv(p2corn, p2csym, true);
            edgei = CubieCubeGetPermSymInv(p2edge, p2esym, false);
        } else if (this->preMoveLen > 0) { // 调整预移动
            int m = UtilStd2ud[lastPre / 3 * 3 + 1]; // 转换为UD轴移动
            this->preMoves[this->preMoveLen - 1] = UtilUd2std[m] * 2 - this->preMoves[this->preMoveLen - 1]; // 调整转动角度
            // 更新坐标
            p2mid = CubieCubeMPermInv[CoordCubeMPermMove[CubieCubeMPermInv[p2mid]][m]];
            p2corn = CoordCubeCPermMove[corni >> 4][CubieCubeSymMoveUD[corni & 0xf][m]];
            corni = (p2corn & ~0xf) | CubieCubeSymMult[p2corn & 0xf][corni & 0xf];
            p2corn = CubieCubeGetPermSymInv(corni >> 4, corni & 0xf, true);
            p2csym = (p2corn & 0xf);
            p2corn >>= 4;
            p2edge = CoordCubeEPermMove[edgei >> 4][CubieCubeSymMoveUD[edgei & 0xf][m]];
            edgei = (p2edge & ~0xf) | CubieCubeSymMult[p2edge & 0xf][edgei & 0xf];
            p2edge = CubieCubeGetPermSymInv(edgei >> 4, edgei & 0xf, false);
            p2esym = p2edge & 0xf;
            p2edge >>= 4;
        }
    }
    // 恢复原始移动
    if (this->depth1 > 0) {
        this->move[this->depth1 - 1] = lastMove;
    }
    if (this->preMoveLen > 0) {
        this->preMoves[this->preMoveLen - 1] = lastPre;
    }
    return ret == 0 ? 0 : 2;
}

/**
 * 阶段1搜索函数
 * @return
 *      0: 找到解法或超过探针限制
 *      1: 尝试增加深度
 *      2: 尝试下一轴
 */
static  int SearchPhase1(Search *this, CoordCube *node, int maxl, int lm) {
    // 阶段1完成且剩余步数小于5，初始化阶段2
    if (node->prun == 0 && maxl < 5) {
        if (this->allowShorter || maxl == 0) {
            this->depth1 -= maxl; // 调整阶段1深度
            int ret = SearchInitPhase2Pre(this); // 初始化阶段2
            this->depth1 += maxl; // 恢复深度
            return ret;
        } else {
            return 1;
        }
    }

    // 遍历18种转动
    for (int axis = 0; axis < 18; axis += 3) {
        if (axis == lm || axis == lm - 9) { // 跳过同面或对立面
            continue;
        }
        for (int power = 0; power < 3; power++) { // 三种转动角度
            int m = axis + power;

            if ((this->isRec && m != this->move[this->depth1 - maxl])) {
                continue;
            }

            // 执行移动并更新坐标
            int prun = CoordCubeDoMovePrun(&this->nodeUD[maxl],node, m, true);
            if (prun > maxl) { // 剪枝
                break;
            } else if (prun == maxl) {
                continue;
            }

            // 共轭剪枝
            prun = CoordCubeDoMovePrunConj(&this->nodeUD[maxl],node, m);
            if (prun > maxl) {
                break;
            } else if (prun == maxl) {
                continue;
            }

            // 记录移动并递归搜索
            this->move[this->depth1 - maxl] = m;
            this->valid1 = min(this->valid1, this->depth1 - maxl); // 更新有效深度
            int ret = SearchPhase1(this,&this->nodeUD[maxl], maxl - 1, axis);
            if (ret == 0) {
                return 0; // 找到解法
            } else if (ret >= 2) {
                break; // 尝试下一轴
            }
        }
    }
    return 1; // 未找到解法，增加深度
}

// 阶段1预移动处理
static int SearchPhase1PreMoves(Search *this, int maxl, int lm, CubieCube *cc) {
    this->preMoveLen = this->maxPreMoves - maxl; // 当前预移动长度
    // 检查是否允许当前预移动序列
    if (this->isRec ? this->depth1 == this->length1 - this->preMoveLen
        : (this->preMoveLen == 0 || (0x36FB7 >> lm & 1) == 0)) {
        this->depth1 = this->length1 - this->preMoveLen; // 阶段1深度
        memcpy(&this->phase1Cubie[0], cc, sizeof(CubieCube)); // 设置初始状态
        this->allowShorter = this->depth1 == MIN_P1LENGTH_PRE && this->preMoveLen != 0; // 是否允许更短解法
        // 设置坐标并剪枝
        if (CoordCubeSetWithPrun(&(this->nodeUD[this->depth1 + 1]), cc, this->depth1)
                && SearchPhase1(this, &(this->nodeUD[this->depth1 + 1]), this->depth1, -1) == 0) {
            return 0; // 找到解法
        }
    }

    // 终止条件：无预移动或阶段1深度不足
    if (maxl == 0 || this->preMoveLen + MIN_P1LENGTH_PRE >= this->length1) {
        return 1;
    }

    int skipMoves = 0;
    // 如果是最后一步预移动，跳过某些转动
    if (maxl == 1 || this->preMoveLen + 1 + MIN_P1LENGTH_PRE >= this->length1) { 
        skipMoves |= 0x36FB7; // 跳过特定移动
    }

    lm = lm / 3 * 3; // 转换为面索引
    // 遍历18种转动作为预移动
    for (int m = 0; m < 18; m++) {
        if (m == lm || m == lm - 9 || m == lm + 9) { // 跳过同面或对立面
            m += 2;
            continue;
        }
        // 跳过禁止移动
        if ((this->isRec && m != this->preMoves[this->maxPreMoves - maxl]) || (skipMoves & 1 << m) != 0) {
            continue;
        }
        // 执行预移动
        CubieCubeCornMult(&CubieCubeMoveCube[m], cc, &(this->preMoveCubes[maxl]));
        CubieCubeEdgeMult(&CubieCubeMoveCube[m], cc, &(this->preMoveCubes[maxl]));
        this->preMoves[this->maxPreMoves - maxl] = m; // 记录预移动
        // 递归处理剩余预移动
        int ret = SearchPhase1PreMoves(this,maxl - 1, m, &this->preMoveCubes[maxl]);
        if (ret == 0) {
            return 0; // 找到解法
        }
    }
    return 1; // 未找到解法
}

// 主搜索函数
static int SearchSearch(Search *this, char *solution_buffer) {
    // 遍历不同解法长度
    for (this->length1 = this->isRec ? this->length1 : 0; this->length1 < this->solLen; this->length1++) {
        this->maxDep2 = min(MAX_DEPTH2, this->solLen - this->length1 - 1); // 阶段2最大深度
        // 遍历6种URF变换
        for (this->urfIdx = this->isRec ? this->urfIdx : 0; this->urfIdx < 6; this->urfIdx++) {
            if (SearchPhase1PreMoves(this,this->maxPreMoves, -30, &this->urfCubieCube[this->urfIdx]) == 0) {
                if(this->solution.initOK == false){
                    // solution_buffer[0] = '\0'; // 无解法
                    // printf("超过探针限制，给定probeMax内无解\n");
                    return -1;
                }else{
                    SolutionToString(&this->solution, solution_buffer); // 输出解法
                    return this->solution.length;
                }
            }
        }
    }
    // 输出最终结果
    if(this->solution.initOK == false){
        // ESP_LOGE(TAG, "给定最大深度内无解");
        return -2;
    }else{
        SolutionToString(&this->solution, solution_buffer);
        return this->solution.length;
    }
}

/*
    转换魔方的表示方式：54个面 ---> 20个棱块角块（位置和方向）
    输入：54个字符的面表示
    输出：棱块和角块的位置与方向
    返回值：
        -1: 无效的棱块（颜色组合无效）
        -2: 存在重复的棱块
        0: 棱块翻转问题（已纠正）
        -4: 无效的角块（颜色组合无效）
        -5: 存在重复的角块
        0: 角块旋转问题（已纠正）
        -7: 棱块角块极性错误（奇偶性不一致）
        -8: 中心块排列错误
        0: 无问题
               | U1 U2 U3 |
               | U4 U5 U6 |
               | U7 U8 U9 |
    | L1 L2 L3 | F1 F2 F3 | R1 R2 R3 | B1 B2 B3 |
    | L4 L5 L6 | F4 F5 F6 | R4 R5 R6 | B4 B5 B6 |
    | L7 L8 L9 | F7 F8 F9 | R7 R8 R9 | B7 B8 B9 |
               | D1 D2 D3 |
               | D4 D5 D6 |
               | D7 D8 D9 |
*/
static int cube_from_face_54(CubieCube *c, const char *cube_str)
{    
    // 面片定义
    enum facelets{
        U1 = 0,  U2 = 1,  U3 = 2,  U4 = 3,  U5 = 4,  U6 = 5,  U7 = 6,  U8 = 7,  U9 = 8,
        R1 = 9,  R2 = 10, R3 = 11, R4 = 12, R5 = 13, R6 = 14, R7 = 15, R8 = 16, R9 = 17,
        F1 = 18, F2 = 19, F3 = 20, F4 = 21, F5 = 22, F6 = 23, F7 = 24, F8 = 25, F9 = 26,
        D1 = 27, D2 = 28, D3 = 29, D4 = 30, D5 = 31, D6 = 32, D7 = 33, D8 = 34, D9 = 35,
        L1 = 36, L2 = 37, L3 = 38, L4 = 39, L5 = 40, L6 = 41, L7 = 42, L8 = 43, L9 = 44,
        B1 = 45, B2 = 46, B3 = 47, B4 = 48, B5 = 49, B6 = 50, B7 = 51, B8 = 52, B9 = 53,
    };
    // 棱块对应的两个面片
    static const char edge_to_face[12][2] = {
        { U6, R2 }, { U8, F2 }, { U4, L2 }, { U2, B2 }, { D6, R8 }, { D2, F8 },
        { D4, L8 }, { D8, B8 }, { F6, R4 }, { F4, L6 }, { B6, L4 }, { B4, R6 }
    };
    // 角块对应的三个面片
    static const char corner_to_face[8][3] = {
        { U9, R1, F3 }, { U7, F1, L3 }, { U1, L1, B3 }, { U3, B1, R3 },
        { D3, F9, R7 }, { D1, L9, F7 }, { D7, B9, L7 }, { D9, R9, B7 }
    };
    // 棱块名称（24种可能方向）
    static const char edge_index[24][2] = {
        "UR","UF","UL","UB","DR","DF","DL","DB","FR","FL","BL","BR",
        "RU","FU","LU","BU","RD","FD","LD","BD","RF","LF","LB","RB"
    };
    // 角块名称（24种可能方向）
    static const char corner_index[24][3] = {
        "URF","UFL","ULB","UBR","DFR","DLF","DBL","DRB",
        "FUR","LUF","BUL","RUB","RDF","FDL","LDB","BDR",
        "RFU","FLU","LBU","BRU","FRD","LFD","BLD","RBD"
    };
    // 检查中心块
    if(cube_str[U5] != 'U' || cube_str[R5] != 'R' || cube_str[F5] != 'F' ||
       cube_str[D5] != 'D' || cube_str[L5] != 'L' || cube_str[B5] != 'B'){
        ESP_LOGE(TAG, "中心块排列错误");
        return -8;
    }
    
    // 处理棱块
    for(int i=0; i<12; i++){
        int index_a = edge_to_face[i][0];
        int index_b = edge_to_face[i][1];
        int tmp;
        // 匹配棱块颜色组合
        for(tmp=0; tmp<24; tmp++){
            if( edge_index[tmp][0] == cube_str[index_a] &&
                edge_index[tmp][1] == cube_str[index_b] ){
                break;
            }
        }
        if(tmp < 24){
            c->ep[i] = tmp % 12; // 位置
            c->er[i] = tmp / 12; // 方向
        }else{
            ESP_LOGE(TAG, "存在无效的棱块%c%c", cube_str[index_a], cube_str[index_b]);
            return -1;
        }
    }
    // 检查棱块位置重复
    for(int i=0; i<12; i++){
        for(int j=0; j<12; j++){
            if(i != j && c->ep[i] == c->ep[j]){
                ESP_LOGE(TAG, "存在重复的棱块");
                return -2;
            }
        }
    }
    // 检查棱块方向奇偶性
    int sum = 0;
    for(int i=0; i<12; i++){
        sum += c->er[i];
    }
    if(sum%2 != 0){
        ESP_LOGE(TAG, "存在棱块翻转问题");
        // 纠正：翻转第一个棱块
        c->er[0] = 1 - c->er[0];
    }
    // 处理角块
    for(int i=0; i<8; i++){
        int index_a = corner_to_face[i][0];
        int index_b = corner_to_face[i][1];
        int index_c = corner_to_face[i][2];
        int tmp;
        // 匹配角块颜色组合
        for(tmp=0; tmp<24; tmp++){
            if( corner_index[tmp][0] == cube_str[index_a] &&
                corner_index[tmp][1] == cube_str[index_b] &&
                corner_index[tmp][2] == cube_str[index_c]){
                break;
            }
        }
        if(tmp < 24){
            c->cp[i] = tmp % 8; // 位置
            c->cr[i] = tmp / 8; // 方向
        }else{
            ESP_LOGE(TAG, "存在无效的角块%c%c%c",
                   cube_str[index_a],cube_str[index_b],cube_str[index_c]);
            return -4;
        }
    }
    // 检查角块位置重复
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(i != j && c->cp[i] == c->cp[j]){
                ESP_LOGE(TAG, "存在重复的角块");
                return -5;
            }
        }
    }
    // 检查角块方向奇偶性（模3）
    sum = 0;
    for(int i=0; i<8; i++){
        sum += c->cr[i];
    }
    if(sum%3 != 0){
        ESP_LOGE(TAG, "存在角块旋转问题");
        // 纠正：调整第一个角块方向
        c->cr[0] = (c->cr[0] + 3 - (sum % 3)) % 3;
    }
    // 校验棱块和角块排列奇偶性
    int cornerParity = 0;
    for (int i = 7; i >= 1; i--){
        for (int j = i - 1; j >= 0; j--){
            if (c->cp[j] > c->cp[i]){
                cornerParity++;
            }
        }
    }
    cornerParity = cornerParity % 2;
    int edgeParity = 0;
    for (int i = 11; i >= 1; i--){
        for (int j = i - 1; j >= 0; j--){
            if (c->ep[j] > c->ep[i]){
                edgeParity++;
            }
        }
    }
    edgeParity = edgeParity % 2;
    if ((edgeParity ^ cornerParity) != 0) {
        ESP_LOGE(TAG, "棱块角块极性错误");
        return -7;
    }
    return 0; // 状态有效
}

// 初始化搜索状态（应用URF变换）
static void SearchInitSearch(Search *this) {
    this -> maxPreMoves = MAX_PRE_MOVES; // 设置最大预移动数
    // 生成6种URF变换状态
    for (int i = 0; i < 6; i++) {
        memcpy(&this -> urfCubieCube[i], &this -> cc, sizeof(CubieCube));
        // 设置坐标状态
        CoordCubeSetWithPrun(&this -> urfCoordCube[i], &this -> urfCubieCube[i], 20);
        CubieCubeURFConjugate(&this -> cc); // 应用URF变换
        if (i % 3 == 2) {
            CubieCubeInvCubieCube(&this -> cc); // 每3次变换取逆
        }
    }

}

// 主求解函数
int SearchSolution(Search *this, char *solution_buffer, const char *facelets, int maxDepth, int probeMax, int probeMin) {
    // 从面表示转换到棱块角块表示
    int check = cube_from_face_54(&this->cc, facelets);

    if (check != 0) {
        solution_buffer[0] = '\0'; // 转换失败
        return check;
    }
    // 初始化搜索参数
    this->solLen = maxDepth + 1;
    this->probe = 0;
    this->probeMax = probeMax;
    this->probeMin = min(probeMin, probeMax);
    memset(&this->solution, 0, sizeof(Solution));
    SearchInitSearch(this); // 初始化搜索状态
    // 执行搜索
    return SearchSearch(this,solution_buffer);;
}

int SearchNext(Search *this, char *solution_buffer, int probeMax, int probeMin) {
    this->probe = 0;
    this->probeMax = probeMax;
    this->probeMin = min(probeMin, probeMax);
    memset(&this->solution, 0, sizeof(Solution));
    this->isRec = true;
    return SearchSearch(this, solution_buffer);
}

/*
static void test_cube_from_face_54_error(void)
{
    const char *test_cases[] = {
        "UUUUUUUUURDRRRRRRRFFFFFFFFFDUDDDDDDDLLLLLLLLLBBBBBBBBB",
        "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLFLLLLLLLBBBBBBBBB",
        "UUULUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLULLLLLLLBBBBBBBBB",
        "UUUUUUUUUURRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
        "UUUUUUUUFLRRRRRRRRFFUFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
        "UUUUUUFUURRRRRRRRRLFFFFFFFFDDDDDDDDDLLULLLLLLBBBBBBBBB",
        "UUUUUUUUURFRRRRRRRFRFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
        "UUUUDUUUURRRRRRRRRFFFFFFFFFDDDDUDDDDLLLLLLLLLBBBBBBBBB",
    };
    for (int i = 0; i < 8; i++) {
        char solution[128] = {0}; 
        printf("测试用例 %d [%s]\n", i + 1, test_cases[i]);
        solve(test_cases[i], solution);
        printf("%s\n",solution);
    }

}
void test_code(void)
{
    test_cube_from_face_54_error();
}
*/