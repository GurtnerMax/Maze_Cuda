#pragma once

#include "EtatChrono.h"

#include <string>

using std::string;
using std::ostream;

class Chrono
    {
    public:

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

	Chrono(const string& title);

	Chrono();

	virtual ~Chrono();

	/*--------------------------------------*\
	|*		Virtual			*|
	 \*-------------------------------------*/

    protected:

	virtual long time()=0;

	virtual double toSeconde(long t)=0;

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	void start();

	/**
	 * [s]
	 */
	double stop();

	/**
	 * [s]
	 */
	double pause();

	/**
	 * after pause
	 */
	void play();

	double getElapseTimeS();

	std::string toString();

	/*--------------------------------------*\
	|*		friend		*|
	 \*-------------------------------------*/

	friend ostream& operator <<(ostream& stream , Chrono& chrono);

	/*--------------------------------------*\
	|*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	string title;

	// Tools
	long timeStart;
	long timeStop;
	long timePause;
	long deltaTime;
	long sumTimeElapsePause;

	EtatChrono etatChrono;
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

