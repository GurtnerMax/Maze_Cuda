#pragma once

#include <math.h>
#include <stdio.h>

#include "Thread2D.cu.h"
#include "Thread1D.cu.h"
#include "cudas.h"

#include "Indices.cu.h"


/*----------------------------------------------------------------------*\
 |*			Tools	 					*|
 \*---------------------------------------------------------------------*/

//	Difficulter:
//
//		(O1)	Partern entrelacement sur la zone centrale (sans if)
//
// Solution:
//
//		(S1) Step A: on fabrique s puis (i,j) de la petiteImage
//	             Step B: on fabrique ss puis (i.j) de la grandeImage
// Definition:
//
//		(D1) zoneCentrale=image-bord ou bord depend fu rayon du kernel de convolution
//
//		(D2) petiteImage= image de base, de taille de la zone centrale, avec le coin en haut a gauche au meme endroit que image de bas
//
//		(D3) petimeImageCentrale = petiteImage translater en i et j de rayon et rayon
//
//		(D4) Grande image = image de base wxh
//
//	Notes:
//
//		(N1) Le resultat de la convolution doit etre calculer que sur petimeImageCentrale
//
//		(N2) petiteImage et petimeImageCentrale sont de meme taille, mais pas eu meme endroit
//
class SousImageIterator
    {
	/*--------------------------------------*\
	|*		Constructor		*|
	 \*-------------------------------------*/

    public:

	__device__ SousImageIterator(uint w , uint h , uint bord) : //
		//
		NB_THREAD(Thread2D::nbThread()), //
		s(Thread2D::tid()), //
		//
		BORD(bord),		//
		//
		w(w),		//
		W_INTERNE((int)(w - 2 * bord)), //
		//H_INTERNE(h - 2 * bord)
		WH_INTERNE(W_INTERNE * (h - 2 * bord)) // W_INTERNE * H_INTERNE
	    {
	    // rien de plus
	    }

	__device__  virtual ~SousImageIterator(void)
	    {
	    // rien
	    }

	/*--------------------------------------*\
	|*		Methodes		*|
	 \*-------------------------------------*/

    public:

	__device__
	int nextS(int* ptrI , int* ptrJ)
	    {
	    int i;
	    int j;

	    // Step A: (i,j,s) petite image:
	    Indices::toIJ(s, W_INTERNE, &i, &j); // on fabrique (i,j) de la petite image

	    // Step B: (i,j,ss) grande image:
	    // Step B.1: on fabrique le (i,j) correspondant de la grande image par translation dans la zone centrale
	    i += BORD;
	    j += BORD;
	    // Step B.2: on fabrique ss de la grande image en partant de (i,j) translater
	    int ss = (i * w) + j;

	    // resumer:
	    //	avant (i,j,s) petiteImage
	    //	apres (i,j,ss) grandeImage

	    s += NB_THREAD;

	    *ptrI = i;
	    *ptrJ = j;
	    return ss;
	    }

	__device__
	bool hasNext()
	    {
	    return s < WH_INTERNE;
	    }

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	int BORD;
	int w;
	int W_INTERNE;
	int WH_INTERNE;

	// Tools
//	int i;
//	int j;
	int s;
	int NB_THREAD;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

