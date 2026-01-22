#pragma once

#include <iostream>
#include "cudas.h"

#include "Maths.h"
#include "Image.h"
#include "Animable_I.h"

#include "MouseKeyListenerGlue_I.h"

class Maze : public Animable_I<uchar4>, public MouseKeyListenerGlue_I
{
public:
    Maze(const Grid& grid, Image& image, bool isVerbose, std::string title="Maze");
    virtual ~Maze(void);

    virtual void process(uchar4* tabPixelsGM, uint w, uint h, const DomaineMath& domaineMath) override;
    virtual void animationStep() override;

    virtual void onMouseMoved(const MouseEvent& event) override;
    virtual void onMousePressed(const MouseEvent& event) override;
    virtual void onMouseReleased(const MouseEvent& event) override;
    virtual void onMouseWheel(const MouseWheelEvent& event) override;

    virtual void onKeyPressed(const KeyEvent& event) override;
    virtual void onKeyReleased(const KeyEvent& event) override;

    bool isStartDefined() const { return hasStart; }
    bool isGoalDefined()  const { return hasGoal;  }

    int getStartX() const { return startHG.x; }
    int getStartY() const { return startHG.y; }
    int getGoalX()  const { return goalHG.x;  }
    int getGoalY()  const { return goalHG.y;  }

private:
    __host__ __device__ int clampi(int v, int lo, int hi) const
    {
        return (v < lo) ? lo : (v > hi) ? hi : v;
    }

    bool isCorridorAt(int x, int y) const;

private:
    Image image;

    // Host image pointer (static)
    uchar4* ptrTabPixelImageHost = nullptr;
    size_t  sizeImage = 0;

    // NEW: device copy of the input image (uploaded once)
    uchar4* ptrImageGM = nullptr;

    // mask
    unsigned char* ptrMaskGM = nullptr;
    unsigned char* ptrMaskTmpGM = nullptr;
    size_t sizeMask = 0;
    bool maskReady = false;

    // auto threshold
    bool autoThresholdDone = false;
    int  binarizeMode = 0;      // 0=luma, 1=exg
    unsigned char seuil = 230;
    int* ptrHistGM = nullptr;

    // start/goal
    bool hasStart = false;
    bool hasGoal  = false;
    int2 startHG = make_int2(-1, -1);
    int2 goalHG  = make_int2(-1, -1);

    int2* ptrStartGM = nullptr;
    int2* ptrGoalGM  = nullptr;

    // solver buffers ping-pong
    int*  ptrLabelA = nullptr;
    int*  ptrLabelB = nullptr;
    unsigned char* ptrDirA = nullptr;
    unsigned char* ptrDirB = nullptr;

    // contact best (GLOBAL, persists across iterations)
    int* ptrBestSumGM      = nullptr; // INT_MAX until found
    int* ptrBestPosIdxGM   = nullptr;
    int* ptrBestNegIdxGM   = nullptr;
    int* ptrBestMeetIdxGM  = nullptr;

    // path + palette helpers
    unsigned char* ptrPathMaskGM = nullptr;
    bool pathReady = false;

    int* ptrMaxAbsGM = nullptr;

    bool solverInit = false;
    bool solverDone = false;

    int iterPerFrame = 32;
};
