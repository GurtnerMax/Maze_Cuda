#pragma once

#include "Chrono.h"

class ChronoSTD: public Chrono
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ChronoSTD(string title = "");

	virtual ~ChronoSTD();

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
	virtual double toSeconde(long microSecondes);

	/*--------------------------------------*\
	 |*		Friend			*|
	 \*-------------------------------------*/

	friend ostream& operator <<(ostream& stream , ChronoSTD& clock);

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

