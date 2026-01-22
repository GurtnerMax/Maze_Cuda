#pragma once

#include "MouseListener_I.h"
#include "KeyListener_I.h"
#include "Maze.h"

class MazeMouseKeyListener : public MouseListener_I, public KeyListener_I
{
public:
    explicit MazeMouseKeyListener(Maze* ptrMaze);
    ~MazeMouseKeyListener() override;

    void onMouseMoved(const MouseEvent& e) override;
    void onMousePressed(const MouseEvent& e) override;
    void onMouseReleased(const MouseEvent& e) override;
    void onMouseWheel(const MouseWheelEvent& e) override;

    void onKeyPressed(const KeyEvent& e) override;
    void onKeyReleased(const KeyEvent& e) override;

private:
    Maze* ptrMaze;
};
