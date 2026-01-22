#pragma once

#include "Chrono.h"

class ChronoOMP: public Chrono
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ChronoOMP(string title = "");

	virtual ~ChronoOMP();

	/*--------------------------------------*\
	|*		Virtual			*|
	 \*-------------------------------------*/

    protected:

	/**
	 * Override
	 */
	virtual long time();

	/**
	 * Override
	 */
	virtual double toSeconde(long timeNano);

	/*--------------------------------------*\
	 |*		Friend			*|
	 \*-------------------------------------*/

	friend ostream& operator <<(ostream& stream , ChronoOMP& clock);

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

