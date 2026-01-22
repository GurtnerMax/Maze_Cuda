#define DISABLE_THIS_FILE 1
#if DISABLE_THIS_FILE
#include <cstdlib>
int mainBenchmark()
{
    return EXIT_SUCCESS; // stub pour le linker
}
#else

//#include <iostream>
//#include <stdlib.h>

#include "MazeProvider.h"

#include "BenchmarkImage.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static void vague();

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainBenchmark()
    {
    cout << "\n[Benchmark] mode" << endl;

    // Attention : Un a la fois seulement!

 //   vague();
//    vagueGray();
//
//    damier();
//    damierRGBAFloat();
//    damierHSBAFloat();
//    damierHueFloat();

    cout << "\n[Benchmark] end" << endl;

    return EXIT_SUCCESS;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

void vague()
    {
   const double DURATION_MAX_S = 8;

    VagueProvider provider;
    BenchmarkImage<uchar4>::run(&provider, DURATION_MAX_S);
    }

/*-----------------------------------*\
 |*		Tools	        	*|
 \*-----------------------------------*/

template<typename T>
void animer(Provider_I<T>* ptrProvider , double durationMaxS , bool isSynchronizeEnable)
    {
    const bool isVerbosityEnable = true;

    Animable_I<T>* ptrAnimable = ptrProvider->createAnimable();

    AnimatorImage<T> animator(ptrAnimable, isVerbosityEnable, isSynchronizeEnable, durationMaxS);
    animator.run();

    delete ptrAnimable;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
#endif
