#include "MazeImageSpecialiser.h"
#include <cmath>

static inline void drawLineThick(Graphic2D& g, int x0, int y0, int x1, int y1, int t)
{
    for (int oy = -t; oy <= t; ++oy)
    {
        const int rx = t - std::abs(oy);
        for (int ox = -rx; ox <= rx; ++ox)
        {
            g.drawLigne(x0 + ox, y0 + oy, x1 + ox, y1 + oy);
        }
    }
}

static void drawCircleThick(Graphic2D& g, int cx, int cy, int r, int n, int t)
{
    const float twoPi = 6.28318530718f;

    float a0 = 0.0f;
    int x0 = cx + (int)(r * cosf(a0));
    int y0 = cy + (int)(r * sinf(a0));

    for (int k = 1; k <= n; ++k)
    {
        const float a1 = twoPi * (float)k / (float)n;
        const int x1 = cx + (int)(r * cosf(a1));
        const int y1 = cy + (int)(r * sinf(a1));

        drawLineThick(g, x0, y0, x1, y1, t);
        x0 = x1; y0 = y1;
    }
}

static void drawCrossXThick(Graphic2D& g, int cx, int cy, int half, int t)
{
    drawLineThick(g, cx - half, cy - half, cx + half, cy + half, t);
    drawLineThick(g, cx - half, cy + half, cx + half, cy - half, t);
}

MazeImageSpecialiser::MazeImageSpecialiser(Maze* ptrMaze, ColorRGB_01 titleColor)
    : ImageAnimable_RGBA_uchar4(ptrMaze, titleColor),
      ptrMaze(ptrMaze)
{
    listener(ptrMaze);
}

MazeImageSpecialiser::~MazeImageSpecialiser()
{
    delete ptrMouseKeyListener;
}

void MazeImageSpecialiser::listener(Maze* ptrMaze)
{
    ptrMouseKeyListener = new MazeMouseKeyListener(ptrMaze);
    setMouseListener(static_cast<MouseListener_I*>(ptrMouseKeyListener));
    setKeyListener(static_cast<KeyListener_I*>(ptrMouseKeyListener));
}

void MazeImageSpecialiser::paintPrimitives(Graphic2D& graphic2D)
{
    ImageAnimable_RGBA_uchar4::paintPrimitives(graphic2D);

    if (ptrMaze->isStartDefined())
    {
        const int x = ptrMaze->getStartX();
        const int y = ptrMaze->getStartY();

        graphic2D.setColorRGB(0.0f, 1.0f, 0.0f);
        drawCircleThick(graphic2D, x, y, 4, 20, 2);

        graphic2D.setFont(TIMES_ROMAN_10);
        graphic2D.drawText(x + 8, y, "Start");
    }

    if (ptrMaze->isGoalDefined())
    {
        const int x = ptrMaze->getGoalX();
        const int y = ptrMaze->getGoalY();

        graphic2D.setColorRGB(1.0f, 0.0f, 0.0f);
        drawCrossXThick(graphic2D, x, y, 4, 2);

        graphic2D.setFont(TIMES_ROMAN_10);
        graphic2D.drawText(x + 8, y, "Goal");
    }
}
