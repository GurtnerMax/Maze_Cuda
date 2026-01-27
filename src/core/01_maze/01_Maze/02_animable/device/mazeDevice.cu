#include "Thread2D.cu.h"
#include "cudas.h"
#include "math/MazeMath.cu.h"

__device__ __constant__ int d_dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
__device__ __constant__ int d_dy8[8] = {-1,-1,-1,  0, 0,  1, 1, 1};

/*---------------- metric ----------------*/
__device__ __forceinline__ unsigned char metricLumaU8(const uchar4& p)
{
    return (unsigned char)((77u * p.x + 150u * p.y + 29u * p.z) >> 8);
}

__device__ __forceinline__ unsigned char metricExGU8(const uchar4& p)
{
    int exg = (int)p.y * 2 - (int)p.x - (int)p.z;
    if (exg < -255) exg = -255;
    if (exg >  510) exg =  510;
    int v = (exg + 255) * 255 / 765;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)v;
}

/*---------------- clear ----------------*/
__global__ void mazeClearUchar(unsigned char* __restrict__ data, uint wh, unsigned char value)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();

    int s = TID;
    while (s < (int)wh)
    {
        data[s] = value;
        s += NB;
    }
}

__global__ void mazeClearInt(int* __restrict__ data, int n, int value)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();

    int i = TID;
    while (i < n)
    {
        data[i] = value;
        i += NB;
    }
}

/*---------------- histogram 256 ----------------*/
__global__ void mazeHistogram256(const uchar4* __restrict__ srcGM,
                                 uint wh,
                                 int* __restrict__ histGM,
                                 int mode)
{
    __shared__ int histS[256];

    int k = threadIdx.x;
    while (k < 256)
    {
        histS[k] = 0;
        k += blockDim.x;
    }
    __syncthreads();

    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();

    int s = TID;
    while (s < (int)wh)
    {
        const uchar4 p = srcGM[s];
        const unsigned char v = (mode == 0) ? metricLumaU8(p) : metricExGU8(p);
        atomicAdd(&histS[(int)v], 1);
        s += NB;
    }
    __syncthreads();

    k = threadIdx.x;
    while (k < 256)
    {
        int v = histS[k];
        if (v) atomicAdd(&histGM[k], v);
        k += blockDim.x;
    }
}

/*---------------- binarize ----------------*/
__global__ void mazeBinarizeMetric(const uchar4* __restrict__ srcGM, uint w, uint h,
                                   unsigned char* __restrict__ maskGM,
                                   unsigned char seuil,
                                   int mode)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();
    const int WH  = (int)(w * h);

    const unsigned char invert = (unsigned char)(mode == 1); // exg: corridor low -> invert

    int s = TID;
    while (s < WH)
    {
        const uchar4 p = srcGM[s];
        const unsigned char v = (mode == 0) ? metricLumaU8(p) : metricExGU8(p);

        const unsigned char pred = (unsigned char)(v > seuil);
        maskGM[s] = (unsigned char)(pred ^ invert);

        s += NB;
    }
}

/*---------------- majority 3x3 (bords gardés) ----------------*/
__global__ void maskMajority3x3(const unsigned char* __restrict__ inMask,
                               unsigned char* __restrict__ outMask,
                               uint w, uint h)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();
    const int WH  = (int)(w * h);
    const int W   = (int)w;
    const int H   = (int)h;

    int s = TID;
    while (s < WH)
    {
        const int i = s / W;
        const int j = s - i * W;

        if (i==0 || j==0 || i==H-1 || j==W-1)
        {
            outMask[s] = inMask[s];
        }
        else
        {
            const int base = s;
            int sum =
                inMask[base - W - 1] + inMask[base - W] + inMask[base - W + 1] +
                inMask[base - 1]     + inMask[base]     + inMask[base + 1] +
                inMask[base + W - 1] + inMask[base + W] + inMask[base + W + 1];

            outMask[s] = (unsigned char)(sum >= 5);
        }

        s += NB;
    }
}

/*---------------- init labels ----------------*/
__global__ void mazeInitLabelsFill(const unsigned char* __restrict__ maskGM,
                                  uint w, uint h,
                                  int* __restrict__ labelGM,
                                  unsigned char* __restrict__ dirGM)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();
    const int WH  = (int)(w * h);

    int s = TID;
    while (s < WH)
    {
        const unsigned char m = maskGM[s];
        labelGM[s] = m ? LABEL_INF : 0;
        dirGM[s]   = 255;
        s += NB;
    }
}

__device__  int stepCostV8(int k)
{
    // si 0 => diag
    const int dx = d_dx8[k];
    const int dy = d_dy8[k];
    return ((dx != 0) & (dy != 0)) ? 3 : 2;  // droit=2, diag=3
}


__global__ void mazeSetSeeds(uint w, int2 start, int2 goal,
                             int* __restrict__ labelGM,
                             unsigned char* __restrict__ dirGM)
{
    if (Thread2D::tid() != 0) return;

    const int W = (int)w;
    const int sStart = start.y * W + start.x;
    const int sGoal  = goal.y  * W + goal.x;

    labelGM[sStart] =  1;
    labelGM[sGoal]  = -1;

    dirGM[sStart] = 254;
    dirGM[sGoal]  = 254;
}

/*---------------- global contact init (ONCE) ----------------*/
__global__ void mazeInitContactGlobal(int* bestSum,
                                      int* bestPosIdx,
                                      int* bestNegIdx,
                                      int* bestMeetIdx)
{
    if (Thread2D::tid() != 0) return;
    *bestSum    = 0x7fffffff;
    *bestPosIdx = -1;
    *bestNegIdx = -1;
    *bestMeetIdx= -1;
}

/*---------------- propagate + detect global best contact ----------------*/
__global__ void mazePropagateV8(const unsigned char* __restrict__ maskGM,
                               const int*  __restrict__ inLabel,
                               const unsigned char* __restrict__ inDir,
                               uint w, uint h,
                               int*  __restrict__ outLabel,
                               unsigned char* __restrict__ outDir,
                               int* __restrict__ bestSum,
                               int* __restrict__ bestPosIdx,
                               int* __restrict__ bestNegIdx,
                               int* __restrict__ bestMeetIdx)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();

    const int W  = (int)w;
    const int H  = (int)h;
    const int WH = W * H;

    int s = TID;
    while (s < WH)
    {
        const int i = s / W;
        const int j = s - i * W;

        // bords: recopie (divergence minime car peu de threads)
        if (i==0 || j==0 || i==H-1 || j==W-1)
        {
            outLabel[s] = inLabel[s];
            outDir[s]   = inDir[s];
            s += NB;
            continue;
        }

        const unsigned char m = maskGM[s];
        const int cur = inLabel[s];

        // mur
        if (!m)
        {
            outLabel[s] = 0;
            outDir[s]   = 255;
            s += NB;
            continue;
        }

        // --- detect contact on labeled cells  ---
        if (cur != LABEL_INF && cur != 0)
        {
            const int absCur = (cur > 0) ? cur : -cur;

            #pragma unroll
            for (int k=0; k<8; k++)
            {
                const int ns = (i + d_dy8[k]) * W + (j + d_dx8[k]);
                const int v  = inLabel[ns];

                if (v == LABEL_INF || v == 0) continue;

                // opposite signs
                if ((cur > 0 && v < 0) || (cur < 0 && v > 0))
                {
                    const int absV = (v > 0) ? v : -v;
                    const int edge = stepCostV8(k);
                    const int sum  = absCur + absV - 2 + edge;

                    const int old = atomicMin(bestSum, sum);
                    if (sum < old)
                    {
                        if (cur > 0)
                        {
                            *bestPosIdx = s;
                            *bestNegIdx = ns;
                        }
                        else
                        {
                            *bestPosIdx = ns;
                            *bestNegIdx = s;
                        }
                        *bestMeetIdx = -1;
                    }
                }
            }

            outLabel[s] = cur;
            outDir[s]   = inDir[s];
            s += NB;
            continue;
        }

        // cchoisi meilleur voisin à prendre (chaque pixel a un poids, 3 diag, 2 droit)
        int bestPosCand = LABEL_INF;     // store best positive for now
        int bestNegCand = -LABEL_INF;    // store curr best negative
        int bestPosS = -1;
        int bestNegS = -1;
        int bestPosK = 255;
        int bestNegK = 255;

        // but=> prendre celui qui a la valeur plus petite et lui rajouter un poids en fonction si diagonal ou droit
        #pragma unroll
        for (int k=0; k<8; k++)
        {
            const int ns = (i + d_dy8[k]) * W + (j + d_dx8[k]);
            //v = current cost pf neighbourg
            const int v  = inLabel[ns];
            if (v == LABEL_INF || v == 0) continue;

            // ajoute une nouvelle valeur pour un voisin= new cost
            const int c = stepCostV8(k);

            if (v > 0)
            {

                const int cand = v + c;
                if (cand < bestPosCand)
                {
                    bestPosCand = cand;
                    bestPosS = ns;
                    bestPosK = k;
                }
                else if (cand == bestPosCand)
                    {
                        // tie break : s best (3) si nouveau droit et ancien était diag
                        if (stepCostV8(k) < stepCostV8(bestPosK))
                        {
                            bestPosS = ns;
                            bestPosK = k;
                        }
                        else if (stepCostV8(k) == stepCostV8(bestPosK) && k < bestPosK)
                        {
                            bestPosS = ns;
                            bestPosK = k;
                        }
                    }
            }

            else // v < 0
            {
                const int cand = v - c;
                // on veut trouver proche de 0
                if (cand > bestNegCand)
                {
                    bestNegCand = cand;
                    bestNegS = ns;
                    bestNegK = k;
                }
                else if (cand == bestNegCand)
                {
                    // tie-break: prefer orth over diag
                    if (stepCostV8(k) < stepCostV8(bestNegK))
                    {
                        bestNegS = ns;
                        bestNegK = k;
                    }
                    else if (stepCostV8(k) == stepCostV8(bestNegK) && k < bestNegK)
                    {
                        bestNegS = ns;
                        bestNegK = k;
                    }
                }
            }
        }

        const int hasPos = (bestPosCand != LABEL_INF);
        const int hasNeg = (bestNegCand != -LABEL_INF);

        // meet-cell contact
        if (hasPos & hasNeg)
        {
            // total cost = (posLabel-1) + (-(negLabel)-1)
            const int sum = (bestPosCand - 1) + ((-bestNegCand) - 1);

            const int old = atomicMin(bestSum, sum);
            if (sum < old)
            {
                *bestPosIdx  = bestPosS;
                *bestNegIdx  = bestNegS;
                *bestMeetIdx = s;
            }
        }

        // propagate
        outLabel[s] = hasPos ? bestPosCand : (hasNeg ? bestNegCand : LABEL_INF);
        outDir[s]   = hasPos ? (unsigned char)bestPosK : (hasNeg ? (unsigned char)bestNegK : 255);


        s += NB;
    }
}


/*---------------- build path mask ----------------*/
__global__ void mazeBuildPathMask(const unsigned char* __restrict__ dirGM,
                                 uint w, uint h,
                                 const int* __restrict__ bestPosIdx,
                                 const int* __restrict__ bestNegIdx,
                                 const int* __restrict__ bestMeetIdx,
                                 unsigned char* __restrict__ pathMaskGM)
{
    if (Thread2D::tid() != 0) return;

    const int W  = (int)w;
    const int H  = (int)h;
    const int WH = W * H;

    const int p0 = *bestPosIdx;
    const int n0 = *bestNegIdx;
    const int m0 = *bestMeetIdx;

    if (p0 < 0 || n0 < 0) return;

    if (m0 >= 0) pathMaskGM[m0] = 1;
    pathMaskGM[p0] = 1;
    pathMaskGM[n0] = 1;

    int s = p0;
    for (int step = 0; step < WH; ++step)
    {
        pathMaskGM[s] = 1;
        const unsigned char d = dirGM[s];
        if (d == 254 || d == 255) break;
        const int k = (int)d;
        s = s + d_dy8[k] * W + d_dx8[k];
        if ((unsigned)s >= (unsigned)WH) break;
    }

    s = n0;
    for (int step = 0; step < WH; ++step)
    {
        pathMaskGM[s] = 1;
        const unsigned char d = dirGM[s];
        if (d == 254 || d == 255) break;
        const int k = (int)d;
        s = s + d_dy8[k] * W + d_dx8[k];
        if ((unsigned)s >= (unsigned)WH) break;
    }
}

/*---------------- maxAbs (warp reduce) ----------------*/
__global__ void mazeResetMaxAbs(int* __restrict__ maxAbsGM)
{
    if (Thread2D::tid() == 0) *maxAbsGM = 1;
}

__device__ __forceinline__ int warpMax(int v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__global__ void mazeComputeMaxAbsLabelWarp(const unsigned char* __restrict__ maskGM,
                                          const int* __restrict__ labelGM,
                                          uint wh,
                                          int* __restrict__ maxAbsGM)
{
    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();

    int local = 1;

    int s = TID;
    while (s < (int)wh)
    {
        if (maskGM[s])
        {
            const int L = labelGM[s];
            if (L != LABEL_INF)
            {
                const int a = (L > 0) ? L : -L;
                local = (a > local) ? a : local;
            }
        }
        s += NB;
    }

    // reduce inside warp
    local = warpMax(local);

    // one atomic per warp
    if ((threadIdx.x & 31) == 0)
        atomicMax(maxAbsGM, local);
}

/*---------------- render adaptive palette + path ----------------*/
__global__ void mazeRenderLabelsHSB_PathAdaptive(const unsigned char* __restrict__ maskGM,
                                                const int* __restrict__ labelGM,
                                                const unsigned char* __restrict__ pathMaskGM,
                                                const int* __restrict__ maxAbsGM,
                                                uint w, uint h,
                                                uchar4* __restrict__ dstGM)
{
    __shared__ int maxAbsS;
    if (threadIdx.x == 0) maxAbsS = *maxAbsGM;
    __syncthreads();

    const int maxA = (maxAbsS > 1) ? maxAbsS : 1;

    const int TID = Thread2D::tid();
    const int NB  = Thread2D::nbThread();
    const int WH  = (int)(w * h);

    int s = TID;
    while (s < WH)
    {
        if (pathMaskGM && pathMaskGM[s])
        {
            dstGM[s] = make_uchar4(255, 0, 0, 255);
            s += NB;
            continue;
        }

        if (!maskGM[s])
        {
            dstGM[s] = make_uchar4(0,0,0,255);
            s += NB;
            continue;
        }

        const int L = labelGM[s];
        if (L == LABEL_INF)
        {
            dstGM[s] = make_uchar4(240,240,240,255);
            s += NB;
            continue;
        }

        const int a = (L > 0) ? L : -L;

        float t = (float)a / (float)maxA;
        if (t > 1.f) t = 1.f;
        t = sqrtf(t);

        float hue = (L > 0)
            ? (0.12f * (1.0f - t) + 0.00f * t)   // yellow->red
            : (0.55f * (1.0f - t) + 0.85f * t);  // cyan->magenta

        const float sat = 1.0f;
        const float val = 1.0f - 0.35f * t;

        const float3 rgb = hsv2rgb(hue, sat, val);

        dstGM[s] = make_uchar4((unsigned char)(255.f * rgb.x),
                               (unsigned char)(255.f * rgb.y),
                               (unsigned char)(255.f * rgb.z),
                               255);

        s += NB;
    }
}
