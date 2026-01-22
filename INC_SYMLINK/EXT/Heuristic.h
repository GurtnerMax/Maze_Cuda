#pragma once

#include "cuda_runtime.h"

#include "Grid.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Heuristic
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	static bool check(const dim3& dg , const dim3& db);

	static bool check(const Grid& grid);

	// atome dg db

	static bool checkDG(const dim3& dg);

	static bool checkDB(const dim3& db);

	static bool checkWarp(const dim3& db);

	static bool checkThreadTotal(const dim3& dg , const dim3& db);

	// atome grid

	static bool checkDG(const Grid& grid);

	static bool checkDB(const Grid& grid);

	static bool checkWarp(const Grid& grid);

	static bool checkThreadTotal(const Grid& grid);

    private:

	static long dim(const dim3& dim);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
