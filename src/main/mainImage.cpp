#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "cudas.h"

#include "MazeProvider.h"

#include "Args.h"
#include "Viewer.h"

using namespace gpu;  // pour Viewer<> et GLUTImageViewers

using std::cout;
using std::endl;
using std::string;

/*----------------------------------------------------------------------*\
 |*                         Declarations                                 *|
 \*---------------------------------------------------------------------*/

int mainImage(const Args& args);

static int mainImageSimple(const Args& args);
// static int mainImageAdvanced(const Args& args);

/*----------------------------------------------------------------------*\
 |*                         Implementation                               *|
 \*---------------------------------------------------------------------*/

int mainImage(const Args& args)
{
    return mainImageSimple(args);
    // return mainImageAdvanced(args);
}

static int mainImageSimple(const Args& args)
{
    cout << "\n[Image] mode" << endl;

    GLUTImageViewers::init(args.argc, args.argv); // only once

    // ImageOption : (isSelection , isAnimation , isOverlay , isShowHelp)
    ImageOption zoomable(true,  true,  false, true);
    ImageOption nozoomable(false, true, false, true);

    // Un seul viewer : affichage de l?image du labyrinthe en niveaux de gris
    Viewer<MazeProvider> maze(nozoomable); // position par défaut

    // Lance toutes les fenêtres (bloquant tant qu?une fenêtre est ouverte)
    GLUTImageViewers::runALL();

    cout << "\n[Image] end" << endl;
    return EXIT_SUCCESS;
}

/* Optionnel si tu veux positionner/redimensionner précisément la fenêtre
static int mainImageAdvanced(const Args& args)
{
    cout << "\n[Image] mode" << endl;

    GLUTImageViewers::init(args.argc, args.argv);

    ImageOption nozoomable(false, true, false, true);

    Viewer<MazeProvider> maze(nozoomable, 0, 0); // px, py
    int w = 1024;
    int h = 512;

    maze.setViewerSize(w, h);
    maze.setViewerPosition(50, 50);

    GLUTImageViewers::runALL();
    cout << "\n[Image] end" << endl;
    return EXIT_SUCCESS;
}
*/
/*----------------------------------------------------------------------*\
 |*                         End                                          *|
 \*---------------------------------------------------------------------*/
