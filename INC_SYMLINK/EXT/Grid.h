#pragma once

#include "cuda_runtime.h"
#include <iostream>

using std::ostream;
using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class Grid
    {
	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	Grid(dim3 dg , dim3 db , bool isCheckHeuristic = true);

	Grid();

	Grid(const Grid& source);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	long threadCounts() const;

	long blockCounts() const;

	long threadByBlock() const;

	/**
	 * grid et block
	 */
	bool is1D() const;

	/**
	 * grid et block
	 */
	bool is2D() const;

	bool isHeuristicDG() const;

	bool isHeuristicDB() const;

	bool isHeuristic() const;

	string toString() const;

	string toStringLight() const;

	static void enableHeuristicCheck(bool isEnable);

	/*--------------*\
	|* friend      *|
	 \*-------------*/

	friend ostream& operator<<(ostream& stream , const Grid& grid);

	/*--------------*\
	|* static      *|
	 \*-------------*/

	static void heuristic(const dim3& dg , const dim3& db);
	static void heuristic(const Grid& grid);

	static void assertion(const dim3& dg , const dim3& db);
	static void assertion(const Grid& grid);

	static long nbThread(const dim3& dg , const dim3& db);
	static long nbThread(const Grid& grid);

	static void print(const dim3& dg , const dim3& db);
	static void print(const Grid& grid);
	static void print(const dim3& dim , string titre = "");

	static long dim(const dim3& dim);


	static void enableDefaultGridCheck(bool isEnable);

private :

	void checkAndAdapt(dim3* prtDg , dim3* ptrDB);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    public:

	// Input
	dim3 dg;
	dim3 db;

    private :

	static bool isEnableDefaultGridCheck;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
