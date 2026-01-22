#pragma once

#include "Indices.cu.h"

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class Edge
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ Edge(int w , int h , int margin) :
		w(w), //
		h(h), //
		margin(margin)
	    {
	    // rien
	    }

	__device__
	   virtual ~Edge()
	    {
	    // rien
	    }

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * s in [0,wh[
	 */
	__device__
	bool isInEdge(int s)
	    {
	    int i;
	    int j;
	    Indices::toIJ(s, w, &i, &j);

	    return isInEdge(i, j);
	    }

	/**
	 * i in [0,h[
	 * j in [0,w[
	 */
	__device__
	bool isInEdge(int i , int j)
	    {
	    return i < margin || j < margin || i > h - margin || j > w - margin;
	    }

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int w;
	int h;
	int margin;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
