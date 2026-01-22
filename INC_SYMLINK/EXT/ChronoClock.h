#pragma once

#include "Chrono.h"

class ChronoClock: public Chrono
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ChronoClock( string title = "");

	virtual ~ChronoClock();

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
	virtual double toSeconde(long clockCount);

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	long getClockCount();

	/*--------------------------------------*\
	 |*		Friend			*|
	 \*-------------------------------------*/

	friend ostream& operator <<(ostream& stream , ChronoClock& clock);

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	/**
	 * Depend of hardware
	 * cuda3 : 1000000
	 */
	static const long CLK_TCK = CLOCKS_PER_SEC;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

