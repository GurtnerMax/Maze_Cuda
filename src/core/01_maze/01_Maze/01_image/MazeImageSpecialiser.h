#pragma once

#include "ImageFromAnimable.h"
#include "MazeMouseKeyListener.h"
#include "Maze.h"

class MazeImageSpecialiser : public ImageAnimable_RGBA_uchar4
{
public:
    MazeImageSpecialiser(Maze* ptrMaze, ColorRGB_01 titleColor);
    ~MazeImageSpecialiser() override;

    void paintPrimitives(Graphic2D& graphic2D) override;

private:
    void listener(Maze* ptrMaze);

private:
    Maze* ptrMaze = nullptr;
    MazeMouseKeyListener* ptrMouseKeyListener = nullptr;
};
