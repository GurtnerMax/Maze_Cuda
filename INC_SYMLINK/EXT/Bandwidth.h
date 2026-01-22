#pragma once

#include <iostream>
#include <string.h>

#include "Chrono.h"

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

class Bandwidth
    {
	/*------------------------------------------------------------*\
        |*			Constructors 			      *|
	 \*------------------------------------------------------------*/

    public:

	/**
	 * start automaticly
	 */
	Bandwidth(size_t sizeOctet , std::string title = "");

	virtual ~Bandwidth();

	/*------------------------------------------------------------*\
        |*			Methodes 			      *|
	 \*------------------------------------------------------------*/

    public:

	void start();
	void stop();

	/**
	 * [MO/s]
	 */
	int getMOS();

	/**
	 * [GO/s]
	 */
	int getGOS();

	string toString();

	friend ostream& operator <<(ostream& stream , Bandwidth& bandwidth);

    private:

	void process(double secondes);

	/*------------------------------------------------------------*\
        |*			Attributs			      *|
	 \*------------------------------------------------------------*/

    private:

	// Input
	size_t sizeOctet;
	std::string title;

	// Tools
	Chrono* ptrChrono;
	double secondes;

	// Output
	double bandwidthMOS;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
