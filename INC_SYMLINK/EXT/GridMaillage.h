#pragma once

#include "Grid.h"
#include "Iterator.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class GridMaillage
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    private:

	GridMaillage(int n , int m , Grid* tabGrid);

    public:

	GridMaillage(Iterator& iteratorDG , Iterator& iteratorDB);

	GridMaillage(Iterator& iteratorDGx , Iterator& iteratorDGy , Iterator& iteratorDBx , Iterator& iteratorDBy);

	GridMaillage(const GridMaillage& source);

	virtual ~GridMaillage();

    private:

	static Grid* clone(Grid* tabGrid , int n);

	/*--------------------------------------*\
	 |*		Metghodes		*|
	 \*-------------------------------------*/

    public:

	Grid* getTabGrid() const;

	int getN() const;

	int getM() const;

	/**
	 * @return n*m
	 */
	int size() const;

	/**
	 * s in [0,nxm[
	 */
	int getNbThread(int s) const;

	/**
	 * i in [0,n[
	 * j in [0,m[
	 */
	int getNbThread(int i , int j) const;

	friend ostream& operator<<(ostream& stream , const GridMaillage& gridMaillage);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int n;
	int m;

	// Outputs
	Grid* tabGrid;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
