#pragma once

#include <iostream>

#include "Grid.h"
#include "cudas.h"

#include "DomaineMath.h"

using std::string;
using std::endl;
using std::cout;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

template<typename T>
class Animable_I
    {

	/*--------------------------------------*\
	    |*		Constructor		*|
	 \*-------------------------------------*/
    public:

	Animable_I(const Grid& grid , unsigned int w , unsigned int h , string title , const DomaineMath& domaineInit , bool isVerbose = true) :
		w(w), //
		h(h), //
		title(title), //
		domaineInit(domaineInit), //
		dg(grid.dg), //
		db(grid.db), //
		isVerbose(isVerbose)
	    {
	    this->t = 0;
	    }

	Animable_I(const Grid& grid , unsigned int w , unsigned int h , string title , bool isVerbose = true) :
		w(w), //
		h(h), //
		title(title), //
		domaineInit(0, w, 0, h), //
		dg(grid.dg), //
		db(grid.db), //
		isVerbose(isVerbose)
	    {
	    this->t = 0;
	    }

	virtual ~Animable_I()
	    {
	    // Nothing here
	    }

	/*--------------------------------------*\
	     |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	virtual void animationStep()=0;

	virtual void process(T* ptrTabPixels , unsigned int w , unsigned int h , const DomaineMath& domaineMath)=0;

	/*--------------------------------------*\
	 |*		Get			*|
	 \*-------------------------------------*/

    public:

	inline float getAnimationPara() const
	    {
	    return t;
	    }

	inline string getTitle() const
	    {
	    return title;
	    }

	inline int getW() const
	    {
	    return w;
	    }

	inline int getH() const
	    {
	    return h;
	    }

	inline DomaineMath getDomaineMathInit() const
	    {
	    return domaineInit;
	    }

	inline dim3 getDg() const
	    {
	    return dg;
	    }

	inline dim3 getDb() const
	    {
	    return db;
	    }

	inline Grid getGrid() const
	    {
	    Grid grid(dg, db);

	    return grid;
	    }

	inline bool isVerbosity() const
	    {
	    return isVerbose;
	    }

	virtual double getInputGO()
	    {
	    // o -> ko -> mo -> go
	    return -1;
	    }

	virtual double getOutputGO()
	    {
	    // o -> ko -> mo -> go
	    return (((w * h * sizeof(T)) / (double)1024) / (double)1024) / (double)1024;
	    }

	/*--------------------------------------*\
	     |*		Attributs		*|
	 \*-------------------------------------*/

    protected:

	// Inputs
	const unsigned int w;
	const unsigned int h;
	const string title;
	DomaineMath domaineInit;
	bool isVerbose;

	//Tools
	double t;
	dim3 dg;
	dim3 db;

    };

typedef Animable_I<uchar4> Animable_I_uchar4;
typedef Animable_I<uchar3> Animable_I_uchar3;
typedef Animable_I<unsigned char> Animable_I_uchar;

typedef Animable_I<float4> Animable_I_float4;
typedef Animable_I<float3> Animable_I_float3;
typedef Animable_I<float> Animable_I_float;

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

