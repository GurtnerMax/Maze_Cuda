#include "MazeMouseKeyListener.h"

MazeMouseKeyListener::MazeMouseKeyListener(Maze* ptrMaze) : ptrMaze(ptrMaze) {}
MazeMouseKeyListener::~MazeMouseKeyListener() = default;

void MazeMouseKeyListener::onMouseMoved(const MouseEvent& e)      { ptrMaze->onMouseMoved(e); }
void MazeMouseKeyListener::onMousePressed(const MouseEvent& e)    { ptrMaze->onMousePressed(e); }
void MazeMouseKeyListener::onMouseReleased(const MouseEvent& e)   { ptrMaze->onMouseReleased(e); }
void MazeMouseKeyListener::onMouseWheel(const MouseWheelEvent& e) { ptrMaze->onMouseWheel(e); }

void MazeMouseKeyListener::onKeyPressed(const KeyEvent& e)        { ptrMaze->onKeyPressed(e); }
void MazeMouseKeyListener::onKeyReleased(const KeyEvent& e)       { ptrMaze->onKeyReleased(e); }
