#pragma once

#include <iostream>
#include <string>

#include "Chrono.h"
#include "ChronoSTD.h"
#include "ChronoOMP.h"
#include "ChronoClock.h"

#include "ChronoType.h"


class ChronoFactory
    {

	/*--------------------------------------*\
	|*		Methode			*|
	 \*-------------------------------------*/

    public:

	/**
	 * <pre>
	 * warning : user need to delete himself ptrChono with
	 *
	 * 		delete ptrChono
	 * </pre>
	 */
	static Chrono* create(ChronoType chronoType , const std::string& title = "");

	/**
	 * <pre>
	 * ChronoType : default is OMP
	 *
	 * warning : user need to delete himself ptrChono with
	 *
	 * 		delete ptrChono
	 * </pre>
	 */
	static Chrono* create(const std::string& title = "");

	/**
	 * <pre>
	 * warning : user need to delete himself ptrChono with
	 *
	 * 		delete ptrChono
	 * </pre>
	 */
	static Chrono* createOMP(const std::string& title = "");

	/**
	 * <pre>
	 * warning : user need to delete himself ptrChono with
	 *
	 * 		delete ptrChono
	 * </pre>
	 */
	static Chrono* createSTD(const std::string& title = "");

	/**
	 * <pre>
	 * warning : user need to delete himself ptrChono with
	 *
	 * 		delete ptrChono
	 * </pre>
	 */
	static Chrono* createClock(const std::string& title = "");

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

