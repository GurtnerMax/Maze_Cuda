#include "MazeProvider.h"
#include "Maze.h"
#include "MazeImageSpecialiser.h"

#include "Grid.h"
#include "Hardware.h"
#include "Image.h"
#include <cassert>

Grid MazeProvider::grid()
{
    const int MP      = Hardware::getMPCount();
    const int CORE_MP = Hardware::getCoreCountMP();

    dim3 dg(MP * 9, 1, 1);
    dim3 db(CORE_MP * 7, 1, 1);
    return Grid(dg, db);
}

Animable_I<uchar4>* MazeProvider::createAnimable(const Grid& grid, bool isVerbose)
{
    const std::string PATH = "/opt/cbi/data/image/maze/";
    std::string name = "maze_real_1024_1024.jpg";

    //const std::string PATH = "/home/mse23/CUDA/toStudent/code/WCudaStudent/Maze_Cuda/";
    //std::string name = "simple_maze.png";

    std::string fileName = PATH + name;

    Image image(fileName);
    return new Maze(grid, image, isVerbose, "Maze");
}

Image_I* MazeProvider::createImageGL()
{
    ColorRGB_01 colorTexte = colorTitle();

    Animable_I<uchar4>* ptrAnimable = Provider_uchar4_A::createAnimable();
    Maze* ptrMaze = dynamic_cast<Maze*>(ptrAnimable);
    assert(ptrMaze != nullptr);

    return new MazeImageSpecialiser(ptrMaze, colorTexte);
}
