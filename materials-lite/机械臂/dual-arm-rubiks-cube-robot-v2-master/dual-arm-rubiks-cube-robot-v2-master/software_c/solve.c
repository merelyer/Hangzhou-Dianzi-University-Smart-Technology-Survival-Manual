#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "table.h"
#include "solve.h"
/*
按照实测10000组随机打乱的魔方，排序访问频率，在MCU上，可将高访问频率的变量放置在RAM，提升性能
sizeof(CubieCubeSym8Move        ) = 576      assces_count = 620437444       count/size = 1077148.4
sizeof(UtilCkmv2bit             ) = 44       assces_count = 18565470        count/size = 421942.5
sizeof(CubieCubeSymMult         ) = 1024     assces_count = 400804788       count/size = 391410.9
sizeof(CubieCubeMoveCubeSym     ) = 144      assces_count = 32585807        count/size = 226290.3
sizeof(CoordCubeMPermMove       ) = 480      assces_count = 100100769       count/size = 208543.3
sizeof(CubieCubeSymMoveUD       ) = 1152     assces_count = 200201538       count/size = 173786.1
sizeof(CubieCubeSymMultInv      ) = 1024     assces_count = 149594506       count/size = 146088.4
sizeof(CoordCubeUDSliceConj     ) = 7920     assces_count = 550693226       count/size = 69532.0
sizeof(CubieCubeFlipS2RF        ) = 5376     assces_count = 319931753       count/size = 59511.1
sizeof(CoordCubeMPermConj       ) = 768      assces_count = 42467631        count/size = 55296.4
sizeof(CubieCubePerm2CombP      ) = 2768     assces_count = 142756164       count/size = 51573.8
sizeof(CubieCubeMoveCube        ) = 360      assces_count = 17926790        count/size = 49796.6
sizeof(CubieCubeCubeSym         ) = 320      assces_count = 13220130        count/size = 41312.9
sizeof(CubieCubePermInvEdgeSym  ) = 5536     assces_count = 200603250       count/size = 36236.1
sizeof(CubieCubeSymMove         ) = 1152     assces_count = 39067428        count/size = 33912.7
sizeof(CoordCubeCCombPConj      ) = 4480     assces_count = 142756164       count/size = 31865.2
sizeof(CoordCubeTwistMove       ) = 11664    assces_count = 310218722       count/size = 26596.3
sizeof(CoordCubeFlipMove        ) = 12096    assces_count = 310218722       count/size = 25646.4
sizeof(UtilCnk                  ) = 676      assces_count = 16781276        count/size = 24824.4
sizeof(CoordCubeUDSliceMove     ) = 17820    assces_count = 271151294       count/size = 15216.1
sizeof(CubieCubeUrf1            ) = 20       assces_count = 240000          count/size = 12000.0
sizeof(CubieCubeUrf2            ) = 20       assces_count = 240000          count/size = 12000.0
sizeof(UtilUd2std               ) = 72       assces_count = 468549          count/size = 6507.6
sizeof(CubieCubeMPermInv        ) = 24       assces_count = 131694          count/size = 5487.2
sizeof(UtilStd2ud               ) = 72       assces_count = 290202          count/size = 4030.6
sizeof(CoordCubeUDSliceTwistPrun) = 80192    assces_count = 275346613       count/size = 3433.6
sizeof(CoordCubeUDSliceFlipPrun ) = 83164    assces_count = 275346613       count/size = 3310.9
sizeof(CubieCubeFlipR2S         ) = 4096     assces_count = 9713031         count/size = 2371.3
sizeof(CubieCubeTwistR2S        ) = 4374     assces_count = 9713031         count/size = 2220.6
sizeof(CubieCubeUrfMove         ) = 108      assces_count = 202548          count/size = 1875.4
sizeof(CoordCubeCPermMove       ) = 55360    assces_count = 100100769       count/size = 1808.2
sizeof(CoordCubeEPermMove       ) = 55360    assces_count = 100100769       count/size = 1808.2
sizeof(CoordCubeMCPermPrun      ) = 33220    assces_count = 42467631        count/size = 1278.4
sizeof(CoordCubeTwistFlipPrun   ) = 331780   assces_count = 319931753       count/size = 964.3
sizeof(CoordCubeEPermCCombPPrun ) = 193764   assces_count = 142756164       count/size = 736.8
sizeof(CubieCubeEPermR2S        ) = 80640    assces_count = 471712          count/size = 5.8
sizeof(CubieCubeFirstMoveSym    ) = 192      assces_count = 0               count/size = 0.0
*/
#ifdef DEBUG_SOLVE
enum COUNT
{
    count_CubieCubeUrf1 = 0,
    count_CubieCubeUrf2,
    count_CubieCubeCubeSym,
    count_CubieCubeMoveCube,
    count_UtilCnk,
    count_UtilUd2std,
    count_UtilStd2ud,
    count_UtilCkmv2bit,
    count_CubieCubeMoveCubeSym,
    count_CubieCubeFirstMoveSym,
    count_CubieCubeSymMult,
    count_CubieCubeSymMultInv,
    count_CubieCubeSymMove,
    count_CubieCubeSym8Move,
    count_CubieCubeSymMoveUD,
    count_CubieCubeUrfMove,
    count_CubieCubePermInvEdgeSym,
    count_CubieCubeFlipR2S,
    count_CubieCubeTwistR2S,
    count_CubieCubeEPermR2S,
    count_CubieCubePerm2CombP,
    count_CubieCubeMPermInv,
    count_CoordCubeUDSliceMove,
    count_CoordCubeTwistMove,
    count_CoordCubeFlipMove,
    count_CoordCubeUDSliceConj,
    count_CoordCubeUDSliceTwistPrun,
    count_CoordCubeUDSliceFlipPrun,
    count_CoordCubeCPermMove,
    count_CoordCubeEPermMove,
    count_CoordCubeMPermMove,
    count_CoordCubeMPermConj,
    count_CoordCubeCCombPConj,
    count_CoordCubeMCPermPrun,
    count_CoordCubeEPermCCombPPrun,
    count_CubieCubeFlipS2RF,
    count_CoordCubeTwistFlipPrun,
};
uint64_t access_count[count_CoordCubeTwistFlipPrun+1] = {0};
#define PRINT_SIZE(x) printf("sizeof(%-25s) = %-8lu assces_count = %-15llu count/size = %.1f\n", #x, sizeof(x), access_count[index], (float)access_count[index]/sizeof(x));index++;
#define DEBUG_COUNT(x) access_count[count_##x]++
void print_table_access(void)
{
    int index = 0;
    PRINT_SIZE(CubieCubeUrf1);
    PRINT_SIZE(CubieCubeUrf2);
    PRINT_SIZE(CubieCubeCubeSym);
    PRINT_SIZE(CubieCubeMoveCube);
    PRINT_SIZE(UtilCnk);
    PRINT_SIZE(UtilUd2std);
    PRINT_SIZE(UtilStd2ud);
    PRINT_SIZE(UtilCkmv2bit);
    PRINT_SIZE(CubieCubeMoveCubeSym);
    PRINT_SIZE(CubieCubeFirstMoveSym);
    PRINT_SIZE(CubieCubeSymMult);
    PRINT_SIZE(CubieCubeSymMultInv);
    PRINT_SIZE(CubieCubeSymMove);
    PRINT_SIZE(CubieCubeSym8Move);
    PRINT_SIZE(CubieCubeSymMoveUD);
    PRINT_SIZE(CubieCubeUrfMove);
    PRINT_SIZE(CubieCubePermInvEdgeSym);
    PRINT_SIZE(CubieCubeFlipR2S);
    PRINT_SIZE(CubieCubeTwistR2S);
    PRINT_SIZE(CubieCubeEPermR2S);
    PRINT_SIZE(CubieCubePerm2CombP);
    PRINT_SIZE(CubieCubeMPermInv);
    PRINT_SIZE(CoordCubeUDSliceMove);
    PRINT_SIZE(CoordCubeTwistMove);
    PRINT_SIZE(CoordCubeFlipMove);
    PRINT_SIZE(CoordCubeUDSliceConj);
    PRINT_SIZE(CoordCubeUDSliceTwistPrun);
    PRINT_SIZE(CoordCubeUDSliceFlipPrun);
    PRINT_SIZE(CoordCubeCPermMove);
    PRINT_SIZE(CoordCubeEPermMove);
    PRINT_SIZE(CoordCubeMPermMove);
    PRINT_SIZE(CoordCubeMPermConj);
    PRINT_SIZE(CoordCubeCCombPConj);
    PRINT_SIZE(CoordCubeMCPermPrun);
    PRINT_SIZE(CoordCubeEPermCCombPPrun);
    PRINT_SIZE(CubieCubeFlipS2RF);
    PRINT_SIZE(CoordCubeTwistFlipPrun);
}
#else
#define DEBUG_COUNT(x)
#endif

static inline int max(int a, int b) {
    return (a > b) ? a : b;
}
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

// -------------------- from Tools.java ------------------------ //
// -------------------- from Util.java ------------------------ //
static char *SolutionToString(Solution *this, char *buffer) {
    char *sb = buffer;
    const char move2str[18][3] = {
        "U ", "U2", "U'", "R ", "R2", "R'", "F ", "F2", "F'",
        "D ", "D2", "D'", "L ", "L2", "L'", "B ", "B2", "B'"
    };
    int urf = (this->verbose & INVERSE_SOLUTION) != 0 ? (this->urfIdx + 3) % 6 : this->urfIdx;
    if (urf < 3) {
        for (int s = 0; s < this->length; s++) {
            if ((this->verbose & USE_SEPARATOR) != 0 && s == this->depth1) {
                strcpy(sb, ".  ");
                sb += 3;
            }
            DEBUG_COUNT(CubieCubeUrfMove);
            strcpy(sb, move2str[CubieCubeUrfMove[urf][this->moves[s]]]);
            sb += 2;
            strcpy(sb, " ");
            sb += 1;
        }
    } else {
        for (int s = this->length - 1; s >= 0; s--) {
            DEBUG_COUNT(CubieCubeUrfMove);
            strcpy(sb, move2str[CubieCubeUrfMove[urf][this->moves[s]]]);
            sb += 2;
            strcpy(sb, " ");
            sb += 1;

            if ((this->verbose & USE_SEPARATOR) != 0 && s == this->depth1) {
                strcpy(sb, ".  ");
                sb += 3;
            }
        }
    }
    if ((this->verbose & APPEND_LENGTH) != 0) {
        sprintf(sb, "(%df)", this->length);
    }
    return buffer;
}

void SolutionSetArgs(Solution *this, int verbose, int urfIdx, int depth1) {
    this->verbose = verbose;
    this->urfIdx = urfIdx;
    this->depth1 = depth1;
}
void SolutionAppendSolMove(Solution *this, int curMove) {
    if (this->length == 0) {
        this->moves[this->length++] = curMove;
        return;
    }
    int axisCur = curMove / 3;
    int axisLast = this->moves[this->length - 1] / 3;
    if (axisCur == axisLast) {
        int pow = (curMove % 3 + this->moves[this->length - 1] % 3 + 1) % 4;
        if (pow == 3) {
            this->length--;
        } else {
            this->moves[this->length - 1] = axisCur * 3 + pow;
        }
        return;
    }
    if (this->length > 1
            && axisCur % 3 == axisLast % 3
            && axisCur == this->moves[this->length - 2] / 3) {
        int pow = (curMove % 3 + this->moves[this->length - 2] % 3 + 1) % 4;
        if (pow == 3) {
            this->moves[this->length - 2] = this->moves[this->length - 1];
            this->length--;
        } else {
            this->moves[this->length - 2] = axisCur * 3 + pow;
        }
        return;
    }
    this->moves[this->length++] = curMove;
}
// 由54个面的颜色表示，转换为8个corner和12个edge表示
static void UtilToCubieCube(CubieCube *ccRet, char *f) 
{
    const int8_t cornerFacelet[8][3] = {
        { U9, R1, F3 }, { U7, F1, L3 }, { U1, L1, B3 }, { U3, B1, R3 },
        { D3, F9, R7 }, { D1, L9, F7 }, { D7, B9, L7 }, { D9, R9, B7 }
    };
    const int8_t edgeFacelet[12][2] = {
        { U6, R2 }, { U8, F2 }, { U4, L2 }, { U2, B2 }, { D6, R8 }, { D2, F8 },
        { D4, L8 }, { D8, B8 }, { F6, R4 }, { F4, L6 }, { B6, L4 }, { B4, R6 }
    };

    int8_t ori;
    for (int i = 0; i < 8; i++) {
        ccRet->ca[i] = 0;
    }
    for (int i = 0; i < 12; i++) {
        ccRet->ea[i] = 0;
    }
    int8_t col1, col2;
    for (int i = 0; i < 8; i++) {
        for (ori = 0; ori < 3; ori++){
            if (f[cornerFacelet[i][ori]] == U || f[cornerFacelet[i][ori]] == D)
                break;
        }
        col1 = f[cornerFacelet[i][(ori + 1) % 3]];
        col2 = f[cornerFacelet[i][(ori + 2) % 3]];
        for (int j = 0; j < 8; j++) {
            if (col1 == cornerFacelet[j][1] / 9 && col2 == cornerFacelet[j][2] / 9) {
                ccRet->ca[i] = (int8_t) (ori % 3 << 3 | j);
                break;
            }
        }
    }
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            if (f[edgeFacelet[i][0]] == edgeFacelet[j][0] / 9
                    && f[edgeFacelet[i][1]] == edgeFacelet[j][1] / 9) {
                ccRet->ea[i] = (int8_t) (j << 1);
                break;
            }
            if (f[edgeFacelet[i][0]] == edgeFacelet[j][1] / 9
                    && f[edgeFacelet[i][1]] == edgeFacelet[j][0] / 9) {
                ccRet->ea[i] = (int8_t) (j << 1 | 1);
                break;
            }
        }
    }
}
static int UtilGetNParity(int idx, int n) 
{
    int p = 0;
    for (int i = n - 2; i >= 0; i--) {
        p ^= idx % (n - i);
        idx /= (n - i);
    }
    return p & 1;
}

static int UtilGetVal(int val0, bool isEdge) {
    return isEdge ? val0 >> 1 : val0 & 7;
}

static int UtilGetNPerm(int8_t arr[], int n, bool isEdge) 
{
    int idx = 0;
    int64_t val = 0xFEDCBA9876543210L;
    for (int i = 0; i < n - 1; i++) {
        int v = UtilGetVal(arr[i], isEdge) << 2;
        idx = (n - i) * idx + (int) (val >> v & 0xf);
        val -= 0x1111111111111110L << v;
    }
    return idx;
}

static int UtilGetComb(int8_t arr[], int arr_length, int mask, bool isEdge) {
    int end = arr_length - 1;
    int idxC = 0, r = 4;
    for (int i = end; i >= 0; i--) {
        int perm = UtilGetVal(arr[i], isEdge);
        if ((perm & 0xc) == mask) {
            DEBUG_COUNT(UtilCnk);
            idxC += UtilCnk[i][r--];
        }
    }
    return idxC;
}

// -------------------- from CubieCube.java ------------------------ //
/*
static void CubieCubeInit(CubieCube *this){
    for(int i = 0; i < 8; i++){
        this->ca[i] = i;
    }
    for(int i = 0; i < 12; i++){
        this->ea[i] = 2*i;
    }
}*/

/**
 * 18 move cubes
 */

/**
 * Notice that Edge Perm Coordnate and Corner Perm Coordnate are the same symmetry structure.
 * So their ClassIndexToRepresentantArray are the same.
 * And when x is RawEdgePermCoordnate, y*16+k is SymEdgePermCoordnate, y*16+(k^e2c[k]) will
 * be the SymCornerPermCoordnate of the State whose RawCornerPermCoordnate is x.
 */
// static byte[] e2c = {0, 0, 0, 0, 1, 3, 1, 3, 1, 3, 1, 3, 0, 0, 0, 0};
static int CubieCubeESym2CSym(int idx) {
    const int SYM_E2C_MAGIC = 0x00DDDD00;
    return idx ^ (SYM_E2C_MAGIC >> ((idx & 0xf) << 1) & 3);
}

// ********************************************* Get and set coordinates *********************************************
// XSym : Symmetry Coordnate of X. MUST be called after initialization of ClassIndexToRepresentantArrays.

// ++++++++++++++++++++ Phase 1 Coordnates ++++++++++++++++++++
// Flip : Orientation of 12 Edges. Raw[0, 2048) Sym[0, 336 * 8)
// Twist : Orientation of 8 Corners. Raw[0, 2187) Sym[0, 324 * 8)
// UDSlice : Positions of the 4 UDSlice edges, the order is ignored. [0, 495)

static int CubieCubeGetFlip(CubieCube *this) {
    int idx = 0;
    for (int i = 0; i < 11; i++) {
        idx = idx << 1 | (this->ea[i] & 1);
    }
    return idx;
}


static int CubieCubeGetFlipSym(CubieCube *this) {
    DEBUG_COUNT(CubieCubeFlipR2S);
    return CubieCubeFlipR2S[CubieCubeGetFlip(this)];
}

static int CubieCubeGetTwist(CubieCube *this) {
    int idx = 0;
    for (int i = 0; i < 7; i++) {
        idx += (idx << 1) + (this->ca[i] >> 3);
    }
    return idx;
}


static int CubieCubeGetTwistSym(CubieCube *this) {
    DEBUG_COUNT(CubieCubeTwistR2S);
    return CubieCubeTwistR2S[CubieCubeGetTwist(this)];
}

static int CubieCubeGetUDSlice(CubieCube *this) {
    return 494 - UtilGetComb(this->ea, 12, 8, true);
}


// ++++++++++++++++++++ Phase 2 Coordnates ++++++++++++++++++++
// EPerm : Permutations of 8 UD Edges. Raw[0, 40320) Sym[0, 2187 * 16)
// Cperm : Permutations of 8 Corners. Raw[0, 40320) Sym[0, 2187 * 16)
// MPerm : Permutations of 4 UDSlice Edges. [0, 24)

static int CubieCubeGetCPerm(CubieCube *this) {
    return UtilGetNPerm(this->ca, 8, false);
}

static int CubieCubeGetCPermSym(CubieCube *this) {
    DEBUG_COUNT(CubieCubeEPermR2S);
    return CubieCubeESym2CSym(CubieCubeEPermR2S[CubieCubeGetCPerm(this)]);
}

static int CubieCubeGetEPerm(CubieCube *this) {
    return UtilGetNPerm(this->ea, 8, true);
}

static int CubieCubeGetEPermSym(CubieCube *this) {
    DEBUG_COUNT(CubieCubeEPermR2S);
    return CubieCubeEPermR2S[CubieCubeGetEPerm(this)];
}

static int CubieCubeGetMPerm(CubieCube *this) {
    return UtilGetNPerm(this->ea, 12, true) % 24;
}

/**
 * Check a cubiecube for solvability. Return the error code.
 * 0: Cube is solvable
 * -2: Not all 12 edges exist exactly once
 * -3: Flip error: One edge has to be flipped
 * -4: Not all corners exist exactly once
 * -5: Twist error: One corner has to be twisted
 * -6: Parity error: Two corners or two edges have to be exchanged
 */
static int CubieCubeVerify(CubieCube *this) 
{
    int sum = 0;
    int edgeMask = 0;
    for (int e = 0; e < 12; e++) {
        edgeMask |= 1 << (this->ea[e] >> 1);
        sum ^= this->ea[e] & 1;
    }
    if (edgeMask != 0xfff) {
        return -2;// missing edges
    }
    if (sum != 0) {
        return -3;
    }
    int cornMask = 0;
    sum = 0;
    for (int c = 0; c < 8; c++) {
        cornMask |= 1 << (this->ca[c] & 7);
        sum += this->ca[c] >> 3;
    }
    if (cornMask != 0xff) {
        return -4;// missing corners
    }
    if (sum % 3 != 0) {
        return -5;// twisted corner
    }
    if ((UtilGetNParity(UtilGetNPerm(this->ea, 12, true), 12) ^ UtilGetNParity(CubieCubeGetCPerm(this), 8)) != 0) {
        return -6;// parity error
    }
    return 0;// cube ok
}

/**
 * prod = a * b, Corner Only.
 */
static void CubieCubeCornMult(const CubieCube *a, const CubieCube *b, CubieCube *prod) {
    for (int corn = 0; corn < 8; corn++) {
        int oriA = a->ca[b->ca[corn] & 7] >> 3;
        int oriB = b->ca[corn] >> 3;
        prod->ca[corn] = (int8_t) ((a->ca[b->ca[corn] & 7] & 7 )| (oriA + oriB) % 3 << 3);
    }
}

/**
 * prod = a * b, Edge Only->
 */
static void CubieCubeEdgeMult(const CubieCube *a, const CubieCube *b, CubieCube *prod) {
    for (int ed = 0; ed < 12; ed++) {
        prod->ea[ed] = (int8_t) (a->ea[b->ea[ed] >> 1] ^ (b->ea[ed] & 1));
    }
}
/**
 * this = S_urf^-1 * this * S_urf.
 */
static void CubieCubeURFConjugate(CubieCube *this) {
    CubieCube temps={0};
    DEBUG_COUNT(CubieCubeUrf2);
    CubieCubeCornMult(&CubieCubeUrf2, this, &temps);
    DEBUG_COUNT(CubieCubeUrf1);
    CubieCubeCornMult(&temps, &CubieCubeUrf1, this);
    DEBUG_COUNT(CubieCubeUrf2);
    CubieCubeEdgeMult(&CubieCubeUrf2, this, &temps);
    DEBUG_COUNT(CubieCubeUrf1);
    CubieCubeEdgeMult(&temps, &CubieCubeUrf1, this);
}
static void CubieCubeInvCubieCube(CubieCube *this) {
    CubieCube temps={0};
    for (int8_t edge = 0; edge < 12; edge++) {
        temps.ea[this->ea[edge] >> 1] = (int8_t) (edge << 1 | (this->ea[edge] & 1));
    }
    for (int8_t corn = 0; corn < 8; corn++) {
        temps.ca[this->ca[corn] & 0x7] = (int8_t) (corn | (0x20 >> (this->ca[corn] >> 3) & 0x18));
    }
    memcpy(this, &temps, sizeof(CubieCube));
}
/**
    * b = S_idx^-1 * a * S_idx, Corner Only.
    */
static void CubieCubeCornConjugate(CubieCube *a, int idx, CubieCube *b) {
    DEBUG_COUNT(CubieCubeCubeSym);
    DEBUG_COUNT(CubieCubeSymMultInv);
    const CubieCube *sinv = &CubieCubeCubeSym[CubieCubeSymMultInv[0][idx]];
    DEBUG_COUNT(CubieCubeCubeSym);
    const CubieCube *s = &CubieCubeCubeSym[idx];
    for (int corn = 0; corn < 8; corn++) {
        int oriA = sinv->ca[a->ca[s->ca[corn] & 7] & 7] >> 3;
        int oriB = a->ca[s->ca[corn] & 7] >> 3;
        int ori = (oriA < 3) ? oriB : (3 - oriB) % 3;
        b->ca[corn] = (int8_t) ((sinv->ca[a->ca[s->ca[corn] & 7] & 7] & 7) | ori << 3);
    }
}

/**
    * b = S_idx^-1 * a * S_idx, Edge Only->
    */
static void CubieCubeEdgeConjugate(CubieCube *a, int idx, CubieCube *b) {
    DEBUG_COUNT(CubieCubeCubeSym);
    DEBUG_COUNT(CubieCubeSymMultInv);
    const CubieCube *sinv = &CubieCubeCubeSym[CubieCubeSymMultInv[0][idx]];
    DEBUG_COUNT(CubieCubeCubeSym);
    const CubieCube *s = &CubieCubeCubeSym[idx];
    for (int ed = 0; ed < 12; ed++) {
        b->ea[ed] = (int8_t) (sinv->ea[a->ea[s->ea[ed] >> 1] >> 1] ^ (a->ea[s->ea[ed] >> 1] & 1) ^ (s->ea[ed] & 1));
    }
}
static int64_t CubieCubeSelfSymmetry(CubieCube *this) {
    CubieCube c;
    memcpy(&c, this, sizeof(CubieCube));
    CubieCube d = {0};
    int cperm = CubieCubeGetCPermSym(this) >> 4;
    int64_t sym = 0L;
    for (int urfInv = 0; urfInv < 6; urfInv++) {
        int cpermx = CubieCubeGetCPermSym(this) >> 4;
        if (cperm == cpermx) {
            for (int i = 0; i < 16; i++) {
                DEBUG_COUNT(CubieCubeSymMultInv);
                CubieCubeCornConjugate(&c, CubieCubeSymMultInv[0][i], &d);
                if (memcmp(d.ca, this->ca, 8) == 0) {
                    DEBUG_COUNT(CubieCubeSymMultInv);
                    CubieCubeEdgeConjugate(&c, CubieCubeSymMultInv[0][i], &d);
                    if (memcmp(d.ea, this->ea, 8) == 0) {
                        sym |= 1L << min(urfInv << 4 | i, 48);
                    }
                }
            }
        }
        CubieCubeURFConjugate(this);
        if (urfInv % 3 == 2) {
            CubieCubeInvCubieCube(this);
        }
    }
    return sym;
}
static int CubieCubeGetPermSymInv(int idx, int sym, bool isCorner) {
    DEBUG_COUNT(CubieCubePermInvEdgeSym);
    int idxi = CubieCubePermInvEdgeSym[idx];
    if (isCorner) {
        idxi = CubieCubeESym2CSym(idxi);
    }
    DEBUG_COUNT(CubieCubeSymMult);
    return (idxi & 0xfff0) | CubieCubeSymMult[idxi & 0xf][sym];
}

static int CubieCubeGetSkipMoves(int64_t ssym) {
    int ret = 0;
    for (int i = 1; (ssym >>= 1) != 0; i++) {
        if ((ssym & 1) == 1) {
            DEBUG_COUNT(CubieCubeFirstMoveSym);
            ret |= CubieCubeFirstMoveSym[i];
        }
    }
    return ret;
}


// -------------------- from CoordCube.java ------------------------ //
static int CoordCubeGetPruning(const int32_t table[], int index) {
    // 原始的代码在ARM aarch64指令集下正常，在ARM EABI5指令集下大部分场合输出0
    // 右移操作符（>>）通常不允许右移的位数大于操作数的位宽。如果右移的位数超过了变量的位宽，
    // 大多数实现会将位移量对位宽取模，即只移动实际的位数。
    // 这种行为并不是C语言标准所规定的，而是由特定平台的实现决定的
    // 在JAVA中是确定的
    // index << 2 和 (index & 7) << 2，在C语言下是不等价的
    // return table[index >> 3] >> (index << 2) & 0xf; // index << 2 <=> (index & 7) << 2
    return table[index >> 3] >> ((index & 7) << 2) & 0xf; 
}
/**
 * @return pruning value
 */
static int CoordCubeDoMovePrun(CoordCube *this, CoordCube *cc, int m, bool isPhase1) {
    DEBUG_COUNT(CoordCubeUDSliceMove);
    this->slice = CoordCubeUDSliceMove[cc->slice][m];
    DEBUG_COUNT(CubieCubeSym8Move);
    DEBUG_COUNT(CoordCubeFlipMove);
    this->flip = CoordCubeFlipMove[cc->flip][CubieCubeSym8Move[m << 3 | cc->fsym]];
    this->fsym = (this->flip & 7) ^ cc->fsym;
    this->flip >>= 3;
    DEBUG_COUNT(CubieCubeSym8Move);
    DEBUG_COUNT(CoordCubeTwistMove);
    this->twist = CoordCubeTwistMove[cc->twist][CubieCubeSym8Move[m << 3 | cc->tsym]];
    this->tsym = (this->twist & 7) ^ cc->tsym;
    this->twist >>= 3;
    DEBUG_COUNT(CoordCubeUDSliceConj);
    DEBUG_COUNT(CoordCubeUDSliceConj);
    DEBUG_COUNT(CoordCubeUDSliceTwistPrun);
    DEBUG_COUNT(CoordCubeUDSliceFlipPrun);
    DEBUG_COUNT(CubieCubeFlipS2RF);
    DEBUG_COUNT(CoordCubeTwistFlipPrun);
    this->prun = max(
                max(
                    CoordCubeGetPruning(CoordCubeUDSliceTwistPrun,
                                this->twist * N_SLICE + CoordCubeUDSliceConj[this->slice][this->tsym]),
                    CoordCubeGetPruning(CoordCubeUDSliceFlipPrun,
                                this->flip * N_SLICE + CoordCubeUDSliceConj[this->slice][this->fsym])),
                USE_TWIST_FLIP_PRUN ? CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                        this->twist << 11 | CubieCubeFlipS2RF[this->flip << 3 | (this->fsym ^ this->tsym)]) : 0);
    return this->prun;
}

static int CoordCubeDoMovePrunConj(CoordCube *this, CoordCube *cc, int m) {
    DEBUG_COUNT(CubieCubeSymMove);
    m = CubieCubeSymMove[3][m];
    DEBUG_COUNT(CubieCubeSym8Move);
    DEBUG_COUNT(CoordCubeFlipMove);
    this->flipc = CoordCubeFlipMove[cc->flipc >> 3][CubieCubeSym8Move[m << 3 | (cc->flipc & 7)]] ^ (cc->flipc & 7);
    DEBUG_COUNT(CubieCubeSym8Move);
    DEBUG_COUNT(CoordCubeTwistMove);
    this->twistc = CoordCubeTwistMove[cc->twistc >> 3][CubieCubeSym8Move[m << 3 | (cc->twistc & 7)]] ^ (cc->twistc & 7);
    DEBUG_COUNT(CubieCubeFlipS2RF);
    DEBUG_COUNT(CoordCubeTwistFlipPrun);
    return CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                        (this->twistc >> 3) << 11 | CubieCubeFlipS2RF[this->flipc ^ (this->twistc & 7)]);
}
static bool CoordCubeSetWithPrun(CoordCube *this, CubieCube *cc, int depth) {
    this->twist = CubieCubeGetTwistSym(cc);
    this->flip = CubieCubeGetFlipSym(cc);
    this->tsym = this->twist & 7;
    this->twist = this->twist >> 3;
    DEBUG_COUNT(CubieCubeFlipS2RF);
    DEBUG_COUNT(CoordCubeTwistFlipPrun);
    this->prun = USE_TWIST_FLIP_PRUN ? CoordCubeGetPruning(CoordCubeTwistFlipPrun,
            this->twist << 11 | CubieCubeFlipS2RF[this->flip ^ this->tsym]) : 0;
    if (this->prun > depth) {
        return false;
    }

    this->fsym = this->flip & 7;
    this->flip = this->flip >> 3;

    this->slice = CubieCubeGetUDSlice(cc);
    DEBUG_COUNT(CoordCubeUDSliceConj);
    DEBUG_COUNT(CoordCubeUDSliceConj);
    DEBUG_COUNT(CoordCubeUDSliceTwistPrun);
    DEBUG_COUNT(CoordCubeUDSliceFlipPrun);
    this->prun = max(this->prun, max(
                        CoordCubeGetPruning(CoordCubeUDSliceTwistPrun,
                                    this->twist * N_SLICE + CoordCubeUDSliceConj[this->slice][this->tsym]),
                        CoordCubeGetPruning(CoordCubeUDSliceFlipPrun,
                                    this->flip * N_SLICE + CoordCubeUDSliceConj[this->slice][this->fsym])));
    if (this->prun > depth) {
        return false;
    }

    if (USE_CONJ_PRUN) {
        CubieCube pc = {};
        CubieCubeCornConjugate(cc, 1, &pc);
        CubieCubeEdgeConjugate(cc, 1, &pc);
        this->twistc = CubieCubeGetTwistSym(&pc);
        this->flipc = CubieCubeGetFlipSym(&pc);
        DEBUG_COUNT(CubieCubeFlipS2RF);
        DEBUG_COUNT(CoordCubeTwistFlipPrun);
        this->prun = max(this->prun,
                        CoordCubeGetPruning(CoordCubeTwistFlipPrun,
                                    (this->twistc >> 3) << 11 | CubieCubeFlipS2RF[this->flipc ^ (this->twistc & 7)]));
    }
    // printf("CoordCubeSetWithPrun %d %d\n",this->prun, depth);
    // CoordCubePrint(this);
    return this->prun <= depth;
}


// -------------------- from Search.java ------------------------ //
static int indexOf(const char *haystack, char needle) {
    for (int i = 0; haystack[i] != '\0'; ++i) {
        if (haystack[i] == needle) {
            return i;
        }
    }
    return -1;
}

//-1: no solution found
// X: solution with X moves shorter than expectation. Hence, the length of the solution is  depth - X
static int SearchPhase2(Search *this, int edge, int esym, int corn, int csym, int mid, int maxl, int depth, int lm) {
    if (edge == 0 && corn == 0 && mid == 0) {
        return maxl;
    }
    DEBUG_COUNT(UtilCkmv2bit);
    int moveMask = UtilCkmv2bit[lm];
    for (int m = 0; m < 10; m++) {
        if ((moveMask >> m & 1) != 0) {
            m += 0x42 >> m & 3;
            continue;
        }
        DEBUG_COUNT(CoordCubeMPermMove);
        int midx = CoordCubeMPermMove[mid][m];
        DEBUG_COUNT(CubieCubeSymMoveUD);
        DEBUG_COUNT(CoordCubeCPermMove);
        int cornx = CoordCubeCPermMove[corn][CubieCubeSymMoveUD[csym][m]];
        DEBUG_COUNT(CubieCubeSymMult);
        int csymx = CubieCubeSymMult[cornx & 0xf][csym];
        cornx >>= 4;
        DEBUG_COUNT(CubieCubeSymMoveUD);
        DEBUG_COUNT(CoordCubeEPermMove);
        int edgex = CoordCubeEPermMove[edge][CubieCubeSymMoveUD[esym][m]];
        DEBUG_COUNT(CubieCubeSymMult);
        int esymx = CubieCubeSymMult[edgex & 0xf][esym];
        edgex >>= 4;
        int edgei = CubieCubeGetPermSymInv(edgex, esymx, false);
        int corni = CubieCubeGetPermSymInv(cornx, csymx, true);
        DEBUG_COUNT(CubieCubeSymMultInv);
        DEBUG_COUNT(CubieCubePerm2CombP);
        DEBUG_COUNT(CoordCubeCCombPConj);
        DEBUG_COUNT(CoordCubeEPermCCombPPrun);
        int prun = CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        (edgei >> 4) * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[corni >> 4] & 0xff][CubieCubeSymMultInv[edgei & 0xf][corni & 0xf]]);
        if (prun > maxl + 1) {
            return maxl - prun + 1;
        } else if (prun >= maxl) {
            m += 0x42 >> m & 3 & (maxl - prun);
            continue;
        }
        DEBUG_COUNT(CubieCubeSymMultInv);
        DEBUG_COUNT(CubieCubePerm2CombP);
        DEBUG_COUNT(CoordCubeMPermConj);
        DEBUG_COUNT(CoordCubeCCombPConj);
        DEBUG_COUNT(CoordCubeMCPermPrun);
        DEBUG_COUNT(CoordCubeEPermCCombPPrun);
        prun = max(
                    CoordCubeGetPruning(CoordCubeMCPermPrun,
                                        cornx * N_MPERM + CoordCubeMPermConj[midx][csymx]),
                    CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        edgex * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[cornx] & 0xff][CubieCubeSymMultInv[esymx][csymx]]));
        if (prun >= maxl) {
            m += 0x42 >> m & 3 & (maxl - prun);
            continue;
        }
        int ret = SearchPhase2(this, edgex, esymx, cornx, csymx, midx, maxl - 1, depth + 1, m);
        if (ret >= 0) {
            DEBUG_COUNT(UtilUd2std);
            this->move[depth] = UtilUd2std[m];
            return ret;
        }
        if (ret < -2) {
            break;
        }
        if (ret < -1) {
            m += 0x42 >> m & 3;
        }
    }
    return -1;
}

static  int SearchInitPhase2(Search *this, int p2corn, int p2csym, int p2edge, int p2esym, int p2mid, int edgei, int corni) {
    DEBUG_COUNT(CubieCubeSymMultInv);
    DEBUG_COUNT(CubieCubeSymMultInv);
    DEBUG_COUNT(CubieCubePerm2CombP);
    DEBUG_COUNT(CubieCubePerm2CombP);
    DEBUG_COUNT(CoordCubeMPermConj);
    DEBUG_COUNT(CoordCubeCCombPConj);
    DEBUG_COUNT(CoordCubeCCombPConj);
    DEBUG_COUNT(CoordCubeMCPermPrun);
    DEBUG_COUNT(CoordCubeEPermCCombPPrun);
    DEBUG_COUNT(CoordCubeEPermCCombPPrun);
    int prun = max(
                    CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                        (edgei >> 4) * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[corni >> 4] & 0xff][CubieCubeSymMultInv[edgei & 0xf][corni & 0xf]]),
                    max(
                        CoordCubeGetPruning(CoordCubeEPermCCombPPrun,
                                            p2edge * N_COMB + CoordCubeCCombPConj[CubieCubePerm2CombP[p2corn] & 0xff][CubieCubeSymMultInv[p2esym][p2csym]]),
                        CoordCubeGetPruning(CoordCubeMCPermPrun,
                                            p2corn * N_MPERM + CoordCubeMPermConj[p2mid][p2csym])));

    if (prun > this->maxDep2) {
        return prun - this->maxDep2;
    }

    int depth2;
    for (depth2 = this->maxDep2; depth2 >= prun; depth2--) {
        int ret = SearchPhase2(this,p2edge, p2esym, p2corn, p2csym, p2mid, depth2, this->depth1, 10);
        if (ret < 0) {
            break;
        }
        depth2 -= ret;
        this->solLen = 0;
        memset(&this->solution, 0, sizeof(Solution));
        this->solution.initOK = true;
        SolutionSetArgs(&this->solution, this->verbose, this->urfIdx, this->depth1);
        for (int i = 0; i < this->depth1 + depth2; i++) {
            SolutionAppendSolMove(&this->solution, this->move[i]);
        }
        for (int i = this->preMoveLen - 1; i >= 0; i--) {
            SolutionAppendSolMove(&this->solution, this->preMoves[i]);
        }
        this->solLen = this->solution.length;
    }

    if (depth2 != this->maxDep2) { //At least one solution has been found.
        this->maxDep2 = min(MAX_DEPTH2, this->solLen - this->length1 - 1);
        return this->probe >= this->probeMin ? 0 : 1;
    }
    return 1;
}
/**
 * @return
 *      0: Found or Probe limit exceeded
 *      1: at least 1 + maxDep2 moves away, Try next power
 *      2: at least 2 + maxDep2 moves away, Try next axis
 */
static  int SearchInitPhase2Pre(Search *this) {
    this->isRec = false;
    if (this->probe >= (this->solution.initOK == false ? this->probeMax : this->probeMin)) {
        return 0;
    }
    ++this->probe;

    for (int i = this->valid1; i < this->depth1; i++) {
        DEBUG_COUNT(CubieCubeMoveCube);
        CubieCubeCornMult(&this->phase1Cubie[i], &CubieCubeMoveCube[this->move[i]], &this->phase1Cubie[i + 1]);
        DEBUG_COUNT(CubieCubeMoveCube);
        CubieCubeEdgeMult(&this->phase1Cubie[i], &CubieCubeMoveCube[this->move[i]], &this->phase1Cubie[i + 1]);
    }
    this->valid1 = this->depth1;

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
    for (int p2switch = 0, p2switchMask = (1 << p2switchMax) - 1;
            p2switch < p2switchMax; p2switch++) {
        // 0 normal; 1 lastmove; 2 lastmove + premove; 3 premove
        if ((p2switchMask >> p2switch & 1) != 0) {
            p2switchMask &= ~(1 << p2switch);
            ret = SearchInitPhase2(this, p2corn, p2csym, p2edge, p2esym, p2mid, edgei, corni);
            if (ret == 0 || ret > 2) {
                break;
            } else if (ret == 2) {
                p2switchMask &= 0x4 << p2switch; // 0->2; 1=>3; 2=>N/A
            }
        }
        if (p2switchMask == 0) {
            break;
        }
        if ((p2switch & 1) == 0 && this->depth1 > 0) {
            DEBUG_COUNT(UtilStd2ud);
            int m = UtilStd2ud[lastMove / 3 * 3 + 1];
            DEBUG_COUNT(UtilUd2std);
            this->move[this->depth1 - 1] = UtilUd2std[m] * 2 - this->move[this->depth1 - 1];
            DEBUG_COUNT(CoordCubeMPermMove);
            p2mid = CoordCubeMPermMove[p2mid][m];
            DEBUG_COUNT(CubieCubeSymMoveUD);
            DEBUG_COUNT(CoordCubeCPermMove);
            p2corn = CoordCubeCPermMove[p2corn][CubieCubeSymMoveUD[p2csym][m]];
            DEBUG_COUNT(CubieCubeSymMult);
            p2csym = CubieCubeSymMult[p2corn & 0xf][p2csym];
            p2corn >>= 4;
            DEBUG_COUNT(CubieCubeSymMoveUD);
            DEBUG_COUNT(CoordCubeEPermMove);
            p2edge = CoordCubeEPermMove[p2edge][CubieCubeSymMoveUD[p2esym][m]];
            DEBUG_COUNT(CubieCubeSymMult);
            p2esym = CubieCubeSymMult[p2edge & 0xf][p2esym];
            p2edge >>= 4;
            corni = CubieCubeGetPermSymInv(p2corn, p2csym, true);
            edgei = CubieCubeGetPermSymInv(p2edge, p2esym, false);
        } else if (this->preMoveLen > 0) {
            DEBUG_COUNT(UtilStd2ud);
            int m = UtilStd2ud[lastPre / 3 * 3 + 1];
            DEBUG_COUNT(UtilUd2std);
            this->preMoves[this->preMoveLen - 1] = UtilUd2std[m] * 2 - this->preMoves[this->preMoveLen - 1];
            DEBUG_COUNT(CubieCubeMPermInv);
            DEBUG_COUNT(CubieCubeMPermInv);
            DEBUG_COUNT(CoordCubeMPermMove);
            p2mid = CubieCubeMPermInv[CoordCubeMPermMove[CubieCubeMPermInv[p2mid]][m]];
            DEBUG_COUNT(CubieCubeSymMoveUD);
            DEBUG_COUNT(CoordCubeCPermMove);
            p2corn = CoordCubeCPermMove[corni >> 4][CubieCubeSymMoveUD[corni & 0xf][m]];
            DEBUG_COUNT(CubieCubeSymMult);
            corni = (p2corn & ~0xf) | CubieCubeSymMult[p2corn & 0xf][corni & 0xf];
            p2corn = CubieCubeGetPermSymInv(corni >> 4, corni & 0xf, true);
            p2csym = (p2corn & 0xf);
            p2corn >>= 4;
            DEBUG_COUNT(CubieCubeSymMoveUD);
            DEBUG_COUNT(CoordCubeEPermMove);
            p2edge = CoordCubeEPermMove[edgei >> 4][CubieCubeSymMoveUD[edgei & 0xf][m]];
            DEBUG_COUNT(CubieCubeSymMult);
            edgei = (p2edge & ~0xf) | CubieCubeSymMult[p2edge & 0xf][edgei & 0xf];
            p2edge = CubieCubeGetPermSymInv(edgei >> 4, edgei & 0xf, false);
            p2esym = p2edge & 0xf;
            p2edge >>= 4;
        }
    }
    if (this->depth1 > 0) {
        this->move[this->depth1 - 1] = lastMove;
    }
    if (this->preMoveLen > 0) {
        this->preMoves[this->preMoveLen - 1] = lastPre;
    }
    return ret == 0 ? 0 : 2;
}
/**
 * @return
 *      0: Found or Probe limit exceeded
 *      1: Try Next Power
 *      2: Try Next Axis
 */
static  int SearchPhase1(Search *this, CoordCube *node, int ssym, int maxl, int lm) {
    if (node->prun == 0 && maxl < 5) {
        if (this->allowShorter || maxl == 0) {
            this->depth1 -= maxl;
            int ret = SearchInitPhase2Pre(this);
            this->depth1 += maxl;
            return ret;
        } else {
            return 1;
        }
    }

    int skipMoves = CubieCubeGetSkipMoves(ssym);

    for (int axis = 0; axis < 18; axis += 3) {
        if (axis == lm || axis == lm - 9) {
            continue;
        }
        for (int power = 0; power < 3; power++) {
            int m = axis + power;

            if ((this->isRec && m != this->move[this->depth1 - maxl])
                    ||(skipMoves != 0 && (skipMoves & 1 << m) != 0)) {
                continue;
            }

            int prun = CoordCubeDoMovePrun(&this->nodeUD[maxl],node, m, true);
            if (prun > maxl) {
                break;
            } else if (prun == maxl) {
                continue;
            }

            if (USE_CONJ_PRUN) {
                prun = CoordCubeDoMovePrunConj(&this->nodeUD[maxl],node, m);
                if (prun > maxl) {
                    break;
                } else if (prun == maxl) {
                    continue;
                }
            }

            this->move[this->depth1 - maxl] = m;
            this->valid1 = min(this->valid1, this->depth1 - maxl);
            DEBUG_COUNT(CubieCubeMoveCubeSym);
            int ret = SearchPhase1(this,&this->nodeUD[maxl], ssym & (int) CubieCubeMoveCubeSym[m], maxl - 1, axis);
            if (ret == 0) {
                return 0;
            } else if (ret >= 2) {
                break;
            }
        }
    }
    return 1;
}
static int SearchPhase1PreMoves(Search *this, int maxl, int lm, CubieCube *cc, int ssym) {
    this->preMoveLen = this->maxPreMoves - maxl;
    if (this->isRec ? this->depth1 == this->length1 - this->preMoveLen
            : (this->preMoveLen == 0 || (0x36FB7 >> lm & 1) == 0)) {
        this->depth1 = this->length1 - this->preMoveLen;
        memcpy(&this->phase1Cubie[0], cc, sizeof(CubieCube));
        this->allowShorter = this->depth1 == MIN_P1LENGTH_PRE && this->preMoveLen != 0;
        if (CoordCubeSetWithPrun(&(this->nodeUD[this->depth1 + 1]), cc, this->depth1)
                && SearchPhase1(this, &(this->nodeUD[this->depth1 + 1]), ssym, this->depth1, -1) == 0) {
            return 0;
        }
    }

    if (maxl == 0 || this->preMoveLen + MIN_P1LENGTH_PRE >= this->length1) {
        return 1;
    }

    int skipMoves = CubieCubeGetSkipMoves(ssym);
    if (maxl == 1 || this->preMoveLen + 1 + MIN_P1LENGTH_PRE >= this->length1) { //last pre move
        skipMoves |= 0x36FB7; // 11 0110 1111 1011 0111
    }

    lm = lm / 3 * 3;
    for (int m = 0; m < 18; m++) {
        if (m == lm || m == lm - 9 || m == lm + 9) {
            m += 2;
            continue;
        }
        if ((this->isRec && m != this->preMoves[this->maxPreMoves - maxl]) || (skipMoves & 1 << m) != 0) {
            continue;
        }
        DEBUG_COUNT(CubieCubeMoveCube);
        CubieCubeCornMult(&CubieCubeMoveCube[m], cc, &(this->preMoveCubes[maxl]));
        DEBUG_COUNT(CubieCubeMoveCube);
        CubieCubeEdgeMult(&CubieCubeMoveCube[m], cc, &(this->preMoveCubes[maxl]));
        this->preMoves[this->maxPreMoves - maxl] = m;
        DEBUG_COUNT(CubieCubeMoveCubeSym);
        int ret = SearchPhase1PreMoves(this,maxl - 1, m, &this->preMoveCubes[maxl], ssym & (int) CubieCubeMoveCubeSym[m]);
        if (ret == 0) {
            return 0;
        }
    }
    return 1;
}

static char *SearchSearch(Search *this, char *solution_buffer) {
    for (this->length1 = this->isRec ? this->length1 : 0; this->length1 < this->solLen; this->length1++) {
        this->maxDep2 = min(MAX_DEPTH2, this->solLen - this->length1 - 1);
        for (this->urfIdx = this->isRec ? this->urfIdx : 0; this->urfIdx < 6; this->urfIdx++) {
            if ((this->conjMask & 1 << this->urfIdx) != 0) {
                continue;
            }
            if (SearchPhase1PreMoves(this,this->maxPreMoves, -30, &this->urfCubieCube[this->urfIdx], (int) (this->selfSym & 0xffff)) == 0) {
                if(this->solution.initOK == false){
                    sprintf(solution_buffer, "Error 8");
                }else{
                    SolutionToString(&this->solution, solution_buffer);
                }
                return solution_buffer;
            }
        }
    }
    if(this->solution.initOK == false){
        sprintf(solution_buffer, "Error 7");
    }else{
        SolutionToString(&this->solution, solution_buffer);
    }
    return solution_buffer;
}





/* 
    转换魔方的表示方式
    输入DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRFRULBLUFDURRBLBDUDL
    输出300504352152110444513222540135232312104540230115453034 
*/ 
static int SearchVerify(Search *this, const char *facelets) 
{
    int32_t count = 0x000000;
    char f[54 + 1];
    char center[6 + 1];

    center[0] = facelets[U5];
    center[1] = facelets[R5];
    center[2] = facelets[F5];
    center[3] = facelets[D5];
    center[4] = facelets[L5];
    center[5] = facelets[B5];
    center[6] = 0;

    for (int i = 0; i < 54; i++) {
        f[i] = indexOf(center, facelets[i]);
        if (f[i] == -1) {
            return -1;
        }
        count += 1 << (f[i] << 2);
    }
    f[54] = 0;
    if (count != 0x999999) {
        return -1;
    }
    UtilToCubieCube(&this->cc, f);
    return CubieCubeVerify(&this->cc);
}


static void SearchInitSearch(Search *this) {
    this -> conjMask = (TRY_INVERSE ? 0 : 0x38) | (TRY_THREE_AXES ? 0 : 0x36);
    this -> selfSym = CubieCubeSelfSymmetry(&this -> cc);
    this -> conjMask |= (this -> selfSym >> 16 & 0xffff) != 0 ? 0x12 : 0;
    this -> conjMask |= (this -> selfSym >> 32 & 0xffff) != 0 ? 0x24 : 0;
    this -> conjMask |= (this -> selfSym >> 48 & 0xffff) != 0 ? 0x38 : 0;
    this -> selfSym &= 0xffffffffffffL;
    this -> maxPreMoves = this -> conjMask > 7 ? 0 : MAX_PRE_MOVES;
    for (int i = 0; i < 6; i++) {
        memcpy(&this -> urfCubieCube[i], &this -> cc, sizeof(CubieCube));
        CoordCubeSetWithPrun(&this -> urfCoordCube[i], &this -> urfCubieCube[i], 20);
        CubieCubeURFConjugate(&this -> cc);
        if (i % 3 == 2) {
            CubieCubeInvCubieCube(&this -> cc);
        }
    }

}

static char *SearchNext(Search *this, char *solution_buffer, int64_t probeMax, int64_t probeMin, int verbose) {
    this->probe = 0;
    this->probeMax = probeMax;
    this->probeMin = min(probeMin, probeMax);
    memset(&this->solution, 0, sizeof(Solution));
    this->isRec = (this->verbose & OPTIMAL_SOLUTION) == (verbose & OPTIMAL_SOLUTION);
    this->verbose = verbose;
    return SearchSearch(this,solution_buffer);
}

/**
 * Computes the solver string for a given cube.
 *
 * @param facelets
 *      is the cube definition string format.<br>
 * The names of the facelet positions of the cube:
 * <pre>
 *             |************|
 *             |*U1**U2**U3*|
 *             |************|
 *             |*U4**U5**U6*|
 *             |************|
 *             |*U7**U8**U9*|
 *             |************|
 * ************|************|************|************|
 * *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*|
 * ************|************|************|************|
 * *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*|
 * ************|************|************|************|
 * *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*|
 * ************|************|************|************|
 *             |************|
 *             |*D1**D2**D3*|
 *             |************|
 *             |*D4**D5**D6*|
 *             |************|
 *             |*D7**D8**D9*|
 *             |************|
 * </pre>
 * A cube definition string "UBL..." means for example: In position U1 we have the U-color, in position U2 we have the
 * B-color, in position U3 we have the L color etc. For example, the "super flip" state is represented as <br>
 * <pre>UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB</pre>
 * and the state generated by "F U' F2 D' B U R' F' L D' R' U' L U B' D2 R' F U2 D2" can be represented as <br>
 * <pre>FBLLURRFBUUFBRFDDFUULLFRDDLRFBLDRFBLUUBFLBDDBUURRBLDDR</pre>
 * You can also use {@link cs.min2phase.Tools#fromScramble(java.lang.String s)} to convert the scramble string to the
 * cube definition string.
 *
 * @param maxDepth
 *      defines the maximal allowed maneuver length. For random cubes, a maxDepth of 21 usually will return a
 *      solution in less than 0.02 seconds on average. With a maxDepth of 20 it takes about 0.1 seconds on average to find a
 *      solution, but it may take much longer for specific cubes.
 *
 * @param probeMax
 *      defines the maximum number of the probes of phase 2. If it does not return with a solution, it returns with
 *      an error code.
 *
 * @param probeMin
 *      defines the minimum number of the probes of phase 2. So, if a solution is found within given probes, the
 *      computing will continue to find shorter solution(s). Btw, if probeMin > probeMax, probeMin will be set to probeMax.
 *
 * @param verbose
 *      determins the format of the solution(s). see USE_SEPARATOR, INVERSE_SOLUTION, APPEND_LENGTH, OPTIMAL_SOLUTION
 *
 * @return The solution string or an error code:<br>
 *      Error 1: There is not exactly one facelet of each colour<br>
 *      Error 2: Not all 12 edges exist exactly once<br>
 *      Error 3: Flip error: One edge has to be flipped<br>
 *      Error 4: Not all corners exist exactly once<br>
 *      Error 5: Twist error: One corner has to be twisted<br>
 *      Error 6: Parity error: Two corners or two edges have to be exchanged<br>
 *      Error 7: No solution exists for the given maxDepth<br>
 *      Error 8: Probe limit exceeded, no solution within given probMax
 */
const char* ERROR_CODE[8]={
    "Error 1: There is not exactly one facelet of each colour",
    "Error 2: Not all 12 edges exist exactly once",
    "Error 3: Flip error: One edge has to be flipped",
    "Error 4: Not all corners exist exactly once",
    "Error 5: Twist error: One corner has to be twisted",
    "Error 6: Parity error: Two corners or two edges have to be exchanged",
    "Error 7: No solution exists for the given maxDepth",
    "Error 8: Probe limit exceeded, no solution within given probMax",
};
static int SearchSolution(Search *this, char *solution_buffer, const char *facelets, int maxDepth, int64_t probeMax, int64_t probeMin, int verbose) {
    int check = SearchVerify(this, facelets);

    if (check != 0) {
        printf("SearchSolution %s\n", ERROR_CODE[abs(check)-1]);
        solution_buffer[0] = '\0';
        return check;
    }
    this->solLen = maxDepth + 1;
    this->probe = 0;
    this->probeMax = probeMax;
    this->probeMin = min(probeMin, probeMax);
    this->verbose = verbose;
    memset(&this->solution, 0, sizeof(Solution));
    this->isRec = false;
    // 使用预先计算的table，因此不需要CoordCube.Init代码
    // CoordCube.init(false);
    SearchInitSearch(this);
    // OPTIMAL_SOLUTION功能运行时间较长，不适合在单片机上运行，因此没有移植searchopt
    // return (verbose & OPTIMAL_SOLUTION) == 0 ? search() : searchopt();
    // 调用顺序如下
    // search -> phase1PreMoves -> phase1 -> initPhase2Pre -> initPhase2 -> phase2
    SearchSearch(this,solution_buffer);
    return this->solution.length;
}

/*
void demo(void)
{
    const char *facelets = "DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRFRULBLUFDURRBLBDUDL";
    char result[128] = {0}; 
    printf("simpleSolve\n");
    printf("%s\n", facelets);
    memset(&search, 0, sizeof(Search));
    SearchSolution(&search, result, facelets, 21, 100000000, 0, 0);
    printf("%s\n", result);
    // R2 U2 B2 L2 F2 U' L2 R2 B2 R2 D  B2 F  L' F  U2 F' R' D' L2 R' 
    
    printf("outputControl\n");
    memset(&search, 0, sizeof(Search));
    SearchSolution(&search, result, facelets, 21, 100000000, 0, APPEND_LENGTH);
    printf("%s\n", result);
    // R2 U2 B2 L2 F2 U' L2 R2 B2 R2 D  B2 F  L' F  U2 F' R' D' L2 R' (21f)
    memset(&search, 0, sizeof(Search));
    SearchSolution(&search, result, facelets, 21, 100000000, 0, USE_SEPARATOR | INVERSE_SOLUTION);
    printf("%s\n", result);
    // R  L2 D  R  F  U2 F' L  F' .  B2 D' R2 B2 R2 L2 U  F2 L2 B2 U2 R2 

    //Find shorter solutions (try more probes even a solution has already been found)
    //In this example, we try AT LEAST 10000 phase2 probes to find shorter solutions.
    printf("findShorterSolutions\n");
    memset(&search, 0, sizeof(Search));
    SearchSolution(&search, result, facelets, 21, 100000000, 10000, 0);
    printf("%s\n", result);
    // L2 U  D2 R' B  U2 L  F  U  R2 D2 F2 U' L2 U  B  D  R'

    //Continue to find shorter solutions
    printf("continueSearch\n");
    memset(&search, 0, sizeof(Search));
    SearchSolution(&search, result, facelets, 21, 500, 0, 0);
    printf("%s\n", result);
    // R2 U2 B2 L2 F2 U' L2 R2 B2 R2 D  B2 F  L' F  U2 F' R' D' L2 R'

    SearchNext(&search, result, 500, 0, 0);
    printf("%s\n", result);
    // D2 L' D' L2 U  R2 F  B  L  B  D' B2 R2 U' R2 U' F2 R2 U' L2

    SearchNext(&search, result, 500, 0, 0);
    printf("%s\n", result);
    // L' U  B  R2 F' L  F' U2 L  U' B' U2 B  L2 F  U2 R2 L2 B2

    SearchNext(&search, result, 500, 0, 0);
    printf("%s\n", result);
    // Error 8, no solution is found after 500 phase2 probes. Let's try more probes.

    SearchNext(&search, result, 500, 0, 0);
    printf("%s\n", result);
    // L2 U  D2 R' B  U2 L  F  U  R2 D2 F2 U' L2 U  B  D  R'
}
*/
// 对于单片机，放到栈中容易溢出
;
int solve(const char *facelets, char *result)
{
    Search *search = (Search *)malloc(sizeof(Search));
    //printf("分配内存 %.1f KB\n", sizeof(Search)/1024.0f);
    memset(search, 0, sizeof(Search));
    int length = SearchSolution(search, result, facelets, 22, 100000000, 200, 0);
    free(search);
    return length;
}

#include <string.h>  // 用于 memcpy

ConstantsData* create_constants_data() {
    // 分配对齐的内存
    ConstantsData* data = aligned_alloc(16, sizeof(ConstantsData));
    if (!data) return NULL;
    
    // 复制 UtilCnk 表
    memcpy(data->UtilCnk, UtilCnk, sizeof(UtilCnk));
    
    // 复制 UtilUd2std 和 UtilStd2ud
    memcpy(data->UtilUd2std, UtilUd2std, sizeof(UtilUd2std));
    memcpy(data->UtilStd2ud, UtilStd2ud, sizeof(UtilStd2ud));
    
    // 加入 UtilCkmv2bit
    memcpy(data->UtilCkmv2bit, UtilCkmv2bit, sizeof(UtilCkmv2bit));
    
    // 复制 CubieCube 常量
    memcpy(&data->CubieCubeUrf1, &CubieCubeUrf1, sizeof(CubieCube));
    memcpy(&data->CubieCubeUrf2, &CubieCubeUrf2, sizeof(CubieCube));
    memcpy(data->CubieCubeCubeSym, CubieCubeCubeSym, sizeof(CubieCube) * 16);
    memcpy(data->CubieCubeMoveCube, CubieCubeMoveCube, sizeof(CubieCube) * 18);
    
    // 复制对称性相关表
    memcpy(data->CubieCubeFirstMoveSym, CubieCubeFirstMoveSym, sizeof(CubieCubeFirstMoveSym));
    
    // 加入 CubieCubeSymMult
    memcpy(data->CubieCubeSymMult, CubieCubeSymMult, sizeof(CubieCubeSymMult));
    
    memcpy(data->CubieCubeSymMultInv, CubieCubeSymMultInv, sizeof(CubieCubeSymMultInv));
    memcpy(data->CubieCubeSymMove, CubieCubeSymMove, sizeof(CubieCubeSymMove));
    memcpy(data->CubieCubeSymMoveUD, CubieCubeSymMoveUD, sizeof(CubieCubeSymMoveUD));
    
    // 加入 CubieCubeMoveCubeSym
    memcpy(data->CubieCubeMoveCubeSym, CubieCubeMoveCubeSym, sizeof(CubieCubeMoveCubeSym));
    
    // 加入 CubieCubeSym8Move
    memcpy(data->CubieCubeSym8Move, CubieCubeSym8Move, sizeof(CubieCubeSym8Move));
    
    // 复制 URF 变换表
    memcpy(data->CubieCubeUrfMove, CubieCubeUrfMove, sizeof(CubieCubeUrfMove));
    
    // 复制坐标转换表
    memcpy(data->CubieCubePermInvEdgeSym, CubieCubePermInvEdgeSym, sizeof(CubieCubePermInvEdgeSym));
    memcpy(data->CubieCubeFlipR2S, CubieCubeFlipR2S, sizeof(CubieCubeFlipR2S));
    memcpy(data->CubieCubeTwistR2S, CubieCubeTwistR2S, sizeof(CubieCubeTwistR2S));
    memcpy(data->CubieCubeEPermR2S, CubieCubeEPermR2S, sizeof(CubieCubeEPermR2S));
    memcpy(data->CubieCubePerm2CombP, CubieCubePerm2CombP, sizeof(CubieCubePerm2CombP));
    memcpy(data->CubieCubeMPermInv, CubieCubeMPermInv, sizeof(CubieCubeMPermInv));
    memcpy(data->CubieCubeFlipS2RF, CubieCubeFlipS2RF, sizeof(CubieCubeFlipS2RF));
    
    // 复制移动表
    memcpy(data->CoordCubeUDSliceMove, CoordCubeUDSliceMove, sizeof(CoordCubeUDSliceMove));
    memcpy(data->CoordCubeTwistMove, CoordCubeTwistMove, sizeof(CoordCubeTwistMove));
    memcpy(data->CoordCubeFlipMove, CoordCubeFlipMove, sizeof(CoordCubeFlipMove));
    memcpy(data->CoordCubeUDSliceConj, CoordCubeUDSliceConj, sizeof(CoordCubeUDSliceConj));
    
    // 复制剪枝表
    memcpy(data->CoordCubeUDSliceTwistPrun, CoordCubeUDSliceTwistPrun, sizeof(CoordCubeUDSliceTwistPrun));
    memcpy(data->CoordCubeUDSliceFlipPrun, CoordCubeUDSliceFlipPrun, sizeof(CoordCubeUDSliceFlipPrun));
    memcpy(data->CoordCubeCPermMove, CoordCubeCPermMove, sizeof(CoordCubeCPermMove));
    memcpy(data->CoordCubeEPermMove, CoordCubeEPermMove, sizeof(CoordCubeEPermMove));
    memcpy(data->CoordCubeMPermConj, CoordCubeMPermConj, sizeof(CoordCubeMPermConj));
    memcpy(data->CoordCubeCCombPConj, CoordCubeCCombPConj, sizeof(CoordCubeCCombPConj));
    memcpy(data->CoordCubeMCPermPrun, CoordCubeMCPermPrun, sizeof(CoordCubeMCPermPrun));
    memcpy(data->CoordCubeEPermCCombPPrun, CoordCubeEPermCCombPPrun, sizeof(CoordCubeEPermCCombPPrun));
    memcpy(data->CoordCubeTwistFlipPrun, CoordCubeTwistFlipPrun, sizeof(CoordCubeTwistFlipPrun));
    
    // 加入 CoordCubeMPermMove
    memcpy(data->CoordCubeMPermMove, CoordCubeMPermMove, sizeof(CoordCubeMPermMove));
    
    return data;
}
void printConstantsDataOffsets() {
    printf("===== Struct ConstantsData Member Offsets and 4-byte Alignment Check =====\n");
    
    // 宏定义用于简化输出
    #define PRINT_OFFSET(member) \
        printf("%-40s %8zu bytes | %s\n", "Offset of " #member ":", \
               offsetof(ConstantsData, member), \
               (offsetof(ConstantsData, member) % 4 == 0 ? "Aligned" : "NOT Aligned!"))
    
    // 第一部分：Util 部分
    PRINT_OFFSET(UtilCnk);
    PRINT_OFFSET(UtilUd2std);
    PRINT_OFFSET(UtilStd2ud);
    PRINT_OFFSET(UtilCkmv2bit);
    
    // 第二部分：CubieCube 部分
    PRINT_OFFSET(CubieCubeUrf1);
    PRINT_OFFSET(CubieCubeUrf2);
    PRINT_OFFSET(CubieCubeCubeSym);
    PRINT_OFFSET(CubieCubeMoveCube);
    PRINT_OFFSET(CubieCubeFirstMoveSym);
    PRINT_OFFSET(CubieCubeSymMult);
    PRINT_OFFSET(CubieCubeSymMultInv);
    PRINT_OFFSET(CubieCubeSymMove);
    PRINT_OFFSET(CubieCubeSymMoveUD);
    PRINT_OFFSET(CubieCubeMoveCubeSym);
    PRINT_OFFSET(CubieCubeSym8Move);
    PRINT_OFFSET(CubieCubeUrfMove);
    PRINT_OFFSET(CubieCubePermInvEdgeSym);
    PRINT_OFFSET(CubieCubeFlipR2S);
    PRINT_OFFSET(CubieCubeEPermR2S);
    PRINT_OFFSET(CubieCubePerm2CombP);
    PRINT_OFFSET(CubieCubeMPermInv);
    PRINT_OFFSET(CubieCubeFlipS2RF);
    PRINT_OFFSET(CubieCubeTwistR2S);
    
    // 第三部分：CoordCube 部分
    PRINT_OFFSET(CoordCubeUDSliceMove);
    PRINT_OFFSET(CoordCubeTwistMove);
    PRINT_OFFSET(CoordCubeFlipMove);
    PRINT_OFFSET(CoordCubeUDSliceConj);
    PRINT_OFFSET(CoordCubeUDSliceTwistPrun);
    PRINT_OFFSET(CoordCubeUDSliceFlipPrun);
    PRINT_OFFSET(CoordCubeCPermMove);
    PRINT_OFFSET(CoordCubeEPermMove);
    PRINT_OFFSET(CoordCubeMPermMove);
    PRINT_OFFSET(CoordCubeMPermConj);
    PRINT_OFFSET(CoordCubeCCombPConj);
    PRINT_OFFSET(CoordCubeMCPermPrun);
    PRINT_OFFSET(CoordCubeEPermCCombPPrun);
    PRINT_OFFSET(CoordCubeTwistFlipPrun);
    
    // 结构体整体信息
    size_t totalSize = sizeof(ConstantsData);
    printf("\n%-40s %8zu bytes | %s\n", "Total struct size:", 
           totalSize, 
           (totalSize % 4 == 0 ? "Aligned" : "NOT Aligned!"));
    
    printf("=========================================================================\n");
    
    // 清理宏定义
    #undef PRINT_OFFSET
}

// 写入文件
int binary_table() {
    ConstantsData* data = create_constants_data();
    printConstantsDataOffsets();
    FILE* file = fopen("table.bin", "wb");
    if (!file) return false;
    
    size_t written = fwrite(data, sizeof(ConstantsData), 1, file);
    fclose(file);
    free(data);
    return written == 1;
}

// // 从文件加载
// ConstantsData* load_from_file(const char* filename) {
//     FILE* file = fopen(filename, "rb");
//     if (!file) return NULL;
    
//     ConstantsData* data = aligned_alloc(16, sizeof(ConstantsData));
//     if (!data) {
//         fclose(file);
//         return NULL;
//     }
    
//     size_t read = fread(data, sizeof(ConstantsData), 1, file);
//     fclose(file);
    
//     if (read != 1) {
//         free(data);
//         return NULL;
//     }
    
//     return data;
// }

