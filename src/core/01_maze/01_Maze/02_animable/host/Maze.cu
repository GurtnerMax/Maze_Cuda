#include "Maze.h"

#include <iostream>
#include <algorithm>
#include <cassert>

#include "GM.h"
#include "Kernel.h"

// device kernels
extern __global__ void mazeClearUchar(unsigned char* data, uint wh, unsigned char value);
extern __global__ void mazeClearInt(int* data, int n, int value);

extern __global__ void mazeHistogram256(const uchar4* srcGM, uint wh, int* histGM, int mode);
extern __global__ void mazeBinarizeMetric(const uchar4* srcGM, uint w, uint h,
                                          unsigned char* maskGM,
                                          unsigned char seuil,
                                          int mode);

extern __global__ void maskMajority3x3(const unsigned char* inMask, unsigned char* outMask, uint w, uint h);

extern __global__ void mazeInitLabelsFill(const unsigned char* maskGM, uint w, uint h, int* labelGM, unsigned char* dirGM);
extern __global__ void mazeSetSeeds(uint w, int2 start, int2 goal, int* labelGM, unsigned char* dirGM);

extern __global__ void mazeInitContactGlobal(int* bestSum, int* bestPosIdx, int* bestNegIdx, int* bestMeetIdx);

extern __global__ void mazePropagateV8(const unsigned char* maskGM,
                                       const int* inLabel, const unsigned char* inDir,
                                       uint w, uint h,
                                       int* outLabel, unsigned char* outDir,
                                       int* bestSum, int* bestPosIdx, int* bestNegIdx, int* bestMeetIdx);

extern __global__ void mazeBuildPathMask(const unsigned char* dirGM,
                                         uint w, uint h,
                                         const int* bestPosIdx,
                                         const int* bestNegIdx,
                                         const int* bestMeetIdx,
                                         unsigned char* pathMaskGM);

extern __global__ void mazeResetMaxAbs(int* maxAbsGM);
extern __global__ void mazeComputeMaxAbsLabelWarp(const unsigned char* maskGM,
                                                  const int* labelGM,
                                                  uint wh,
                                                  int* maxAbsGM);

extern __global__ void mazeRenderLabelsHSB_PathAdaptive(const unsigned char* maskGM,
                                                        const int* labelGM,
                                                        const unsigned char* pathMaskGM,
                                                        const int* maxAbsGM,
                                                        uint w, uint h,
                                                        uchar4* dstGM);

/*---------------- Otsu (host) ----------------*/
static unsigned char otsuThreshold256(const int hist[256], int total)
{
    long long sumAll = 0;
    for (int i=0;i<256;i++) sumAll += (long long)i * (long long)hist[i];

    long long sumB = 0;
    int wB = 0;

    double bestVar = -1.0;
    int bestT = 128;

    for (int t=0;t<256;t++)
    {
        wB += hist[t];
        if (wB == 0) continue;

        int wF = total - wB;
        if (wF == 0) break;

        sumB += (long long)t * (long long)hist[t];

        const double mB = (double)sumB / (double)wB;
        const double mF = (double)(sumAll - sumB) / (double)wF;

        const double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > bestVar)
        {
            bestVar = varBetween;
            bestT = t;
        }
    }
    return (unsigned char)bestT;
}

static int chooseBinarizeModeFromHost(const uchar4* pixels, int w, int h)
{
    const int stepX = std::max(1, w / 64);
    const int stepY = std::max(1, h / 64);

    long long sumChroma = 0;
    long long n = 0;

    for (int y=0; y<h; y+=stepY)
    {
        for (int x=0; x<w; x+=stepX)
        {
            const uchar4 p = pixels[y*w + x];
            const int mx = std::max({(int)p.x,(int)p.y,(int)p.z});
            const int mn = std::min({(int)p.x,(int)p.y,(int)p.z});
            sumChroma += (mx - mn);
            n++;
        }
    }

    const double avgChroma = (n>0) ? (double)sumChroma / (double)n : 0.0;
    return (avgChroma > 18.0) ? 1 : 0; // 1=exg, 0=luma
}

Maze::Maze(const Grid& grid, Image& image, bool isVerbose, std::string title)
    : Animable_I<uchar4>(grid, image.w(), image.h(), title, isVerbose),
      image(image)
{
    ptrTabPixelImageHost = image.uchar4_RGBA();
    sizeImage = sizeof(uchar4) * w * h;

    sizeMask = (size_t)w * (size_t)h * sizeof(unsigned char);

    // device input image uploaded once
    GM::malloc(&ptrImageGM, sizeImage);
    GM::memcpyHToD(ptrImageGM, ptrTabPixelImageHost, sizeImage);

    // mask
    GM::malloc(&ptrMaskGM,    sizeMask);
    GM::malloc(&ptrMaskTmpGM, sizeMask);

    // start/goal
    GM::malloc(&ptrStartGM, sizeof(int2));
    GM::malloc(&ptrGoalGM,  sizeof(int2));
    GM::memcpyHToD(ptrStartGM, &startHG, sizeof(int2));
    GM::memcpyHToD(ptrGoalGM,  &goalHG,  sizeof(int2));

    // solver buffers
    const size_t wh = (size_t)w * (size_t)h;
    GM::malloc(&ptrLabelA, wh * sizeof(int));
    GM::malloc(&ptrLabelB, wh * sizeof(int));
    GM::malloc(&ptrDirA,   wh * sizeof(unsigned char));
    GM::malloc(&ptrDirB,   wh * sizeof(unsigned char));

    // global best contact
    GM::malloc(&ptrBestSumGM,      sizeof(int));
    GM::malloc(&ptrBestPosIdxGM,   sizeof(int));
    GM::malloc(&ptrBestNegIdxGM,   sizeof(int));
    GM::malloc(&ptrBestMeetIdxGM,  sizeof(int));

    // path
    GM::malloc(&ptrPathMaskGM, wh * sizeof(unsigned char));
    mazeClearUchar<<<dg, db>>>(ptrPathMaskGM, (uint)wh, 0);

    // histogram
    GM::malloc(&ptrHistGM, 256 * sizeof(int));

    // maxAbs scalar
    GM::malloc(&ptrMaxAbsGM, sizeof(int));
}

Maze::~Maze()
{
    GM::free(ptrImageGM);

    GM::free(ptrMaskGM);
    GM::free(ptrMaskTmpGM);

    GM::free(ptrStartGM);
    GM::free(ptrGoalGM);

    GM::free(ptrLabelA);
    GM::free(ptrLabelB);
    GM::free(ptrDirA);
    GM::free(ptrDirB);

    GM::free(ptrBestSumGM);
    GM::free(ptrBestPosIdxGM);
    GM::free(ptrBestNegIdxGM);
    GM::free(ptrBestMeetIdxGM);

    GM::free(ptrPathMaskGM);
    GM::free(ptrHistGM);
    GM::free(ptrMaxAbsGM);

    ptrImageGM = nullptr;
    ptrMaskGM = ptrMaskTmpGM = nullptr;
    ptrStartGM = ptrGoalGM = nullptr;
    ptrLabelA = ptrLabelB = nullptr;
    ptrDirA = ptrDirB = nullptr;
    ptrBestSumGM = ptrBestPosIdxGM = ptrBestNegIdxGM = ptrBestMeetIdxGM = nullptr;
    ptrPathMaskGM = nullptr;
    ptrHistGM = nullptr;
    ptrMaxAbsGM = nullptr;
}

bool Maze::isCorridorAt(int x, int y) const
{
    if (!maskReady) return false;

    cudaDeviceSynchronize();

    const int ix = clampi(x, 0, (int)w - 1);
    const int iy = clampi(y, 0, (int)h - 1);
    const int s  = iy * (int)w + ix;

    unsigned char v = 0;
    GM::memcpyDToH(&v, ptrMaskGM + s, sizeof(unsigned char));
    return v != 0;
}

void Maze::process(uchar4* tabPixelsGM, uint w, uint h, const DomaineMath&)
{
    const uint wh = w * h;

    // 1) mask only once
    if (!maskReady)
    {
        if (!autoThresholdDone)
        {
            binarizeMode = chooseBinarizeModeFromHost(ptrTabPixelImageHost, (int)w, (int)h);

            mazeClearInt<<<dg, db>>>(ptrHistGM, 256, 0);
            mazeHistogram256<<<dg, db>>>(ptrImageGM, wh, ptrHistGM, binarizeMode);

            int histH[256];
            GM::memcpyDToH(histH, ptrHistGM, 256 * sizeof(int));

            seuil = otsuThreshold256(histH, (int)wh);

            std::cout << "\n[AutoThreshold] mode=" << (binarizeMode==0 ? "LUMA" : "EXG")
                      << " seuil=" << (int)seuil << "\n";

            autoThresholdDone = true;
        }

        mazeBinarizeMetric<<<dg, db>>>(ptrImageGM, w, h, ptrMaskTmpGM, seuil, binarizeMode);
        maskMajority3x3<<<dg, db>>>(ptrMaskTmpGM, ptrMaskGM, w, h);

        maskReady = true;
    }

    // 2) solver (only when both points defined)
    if (hasStart && hasGoal && !solverDone)
    {
        if (!solverInit)
        {
            mazeClearUchar<<<dg, db>>>(ptrPathMaskGM, wh, 0);
            pathReady = false;

            mazeInitLabelsFill<<<dg, db>>>(ptrMaskGM, w, h, ptrLabelA, ptrDirA);
            mazeSetSeeds<<<1,1>>>(w, startHG, goalHG, ptrLabelA, ptrDirA);

            mazeInitContactGlobal<<<1,1>>>(ptrBestSumGM, ptrBestPosIdxGM, ptrBestNegIdxGM, ptrBestMeetIdxGM);

            solverInit = true;
            solverDone = false;
        }

        int it = 0;
        while (it < iterPerFrame)
        {
            mazePropagateV8<<<dg, db>>>(ptrMaskGM,
                                        ptrLabelA, ptrDirA,
                                        w, h,
                                        ptrLabelB, ptrDirB,
                                        ptrBestSumGM,
                                        ptrBestPosIdxGM,
                                        ptrBestNegIdxGM,
                                        ptrBestMeetIdxGM);

            std::swap(ptrLabelA, ptrLabelB);
            std::swap(ptrDirA,   ptrDirB);

            it++;
        }

        // one single D->H per frame
        int bestSumH = 0x7fffffff;
        GM::memcpyDToH(&bestSumH, ptrBestSumGM, sizeof(int));
        solverDone = (bestSumH != 0x7fffffff);
    }

    // 3) build path once
    if (solverInit && solverDone && !pathReady)
    {
        mazeClearUchar<<<dg, db>>>(ptrPathMaskGM, wh, 0);

        mazeBuildPathMask<<<1,1>>>(ptrDirA, w, h,
                                   ptrBestPosIdxGM,
                                   ptrBestNegIdxGM,
                                   ptrBestMeetIdxGM,
                                   ptrPathMaskGM);
        pathReady = true;
    }

    // 4) render (always)
    if (solverInit)
    {
        mazeResetMaxAbs<<<1,1>>>(ptrMaxAbsGM);
        mazeComputeMaxAbsLabelWarp<<<dg, db>>>(ptrMaskGM, ptrLabelA, wh, ptrMaxAbsGM);

        mazeRenderLabelsHSB_PathAdaptive<<<dg, db>>>(ptrMaskGM, ptrLabelA, ptrPathMaskGM,
                                                     ptrMaxAbsGM, w, h, tabPixelsGM);
    }
    else
    {
        // no solver yet -> show mask-like neutral rendering
        mazeResetMaxAbs<<<1,1>>>(ptrMaxAbsGM); // harmless
        mazeRenderLabelsHSB_PathAdaptive<<<dg, db>>>(ptrMaskGM, ptrLabelA, nullptr,
                                                     ptrMaxAbsGM, w, h, tabPixelsGM);
    }
}

void Maze::animationStep() {}

void Maze::onMousePressed(const MouseEvent& event)
{
    int x = clampi(event.getX(), 0, (int)w - 1);
    int y = clampi(event.getY(), 0, (int)h - 1);

    if (!isCorridorAt(x, y))
    {
        std::cout << "\n[Rejected] wall at (x,y)=(" << x << "," << y << ")\n";
        return;
    }

    if (!hasStart)
    {
        hasStart = true;
        startHG = make_int2(x, y);
        GM::memcpyHToD(ptrStartGM, &startHG, sizeof(int2));

        solverInit = false;
        solverDone = false;
        pathReady  = false;

        std::cout << "\n[Start] : at pixel (x,y) = (" << x << "," << y << ")\n";
        return;
    }

    if (!hasGoal)
    {
        hasGoal = true;
        goalHG = make_int2(x, y);
        GM::memcpyHToD(ptrGoalGM, &goalHG, sizeof(int2));

        solverInit = false;
        solverDone = false;
        pathReady  = false;

        std::cout << "[Goal]  : at pixel (x,y) = (" << x << "," << y << ")\n";
        return;
    }

    // reset cycle
    hasStart = true;
    hasGoal  = false;

    startHG = make_int2(x, y);
    goalHG  = make_int2(-1, -1);

    GM::memcpyHToD(ptrStartGM, &startHG, sizeof(int2));
    GM::memcpyHToD(ptrGoalGM,  &goalHG,  sizeof(int2));

    solverInit = false;
    solverDone = false;
    pathReady  = false;

    std::cout << "\n[Reset]\n";
    std::cout << "[Start] : at pixel (x,y) = (" << x << "," << y << ")\n";
}

void Maze::onMouseMoved(const MouseEvent&) {}
void Maze::onMouseReleased(const MouseEvent&) {}
void Maze::onMouseWheel(const MouseWheelEvent&) {}
void Maze::onKeyPressed(const KeyEvent&) {}
void Maze::onKeyReleased(const KeyEvent&) {}
