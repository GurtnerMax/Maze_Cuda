#pragma once

#include <iostream>

#include "Runnable_I.h"
#include "Chrono.h"

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/**
 * <pre>
 * chrono pas assez fin pour calculer le temps de 1 iteration precisemment!
 * On fait des lots ou la taille est detecter automatiqument pour que le chono puisse chronometrer avec assez de precision
 * </pre>
 */
class FpsCalculator
    {
	/*--------------------------------------*\
	    |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	FpsCalculator(Runnable_I* ptrRunnable , bool isVerboseEnable = true , double nbSecondesProcessingMax = 8);

	FpsCalculator(Runnable_I* ptrRunnable , double nbSecondesProcessingMax);

	virtual ~FpsCalculator();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * return fpsMedian
	 */
	long run();

	void print();

	/*--------------*\
	 |*	get	*|
	 \*-------------*/

	long getFpsGlobal();

	long getFpsMean();
	long getFpsMedian();
	long getFpsMin();
	long getFpsMax();
	long getFpsStd();

	//double getStd();
	double getPourcentageErreur();

	long getNbIterationBylot();
	long getNbIterationTotal();
	long getNLot();

	//double getGOsInputGlobal();
	//double getGOsInputMean();
	double getGOsInputMedian();

	//double getGOsOutputGlobal();
	//double getGOsOutputMean();
	double getGOsOutputMedian();

//	double getDurationGlobalIteration1_MS();
//	double getDurationMeanIteration1_MS();
//	double getDurationMedianIteration1_MS();

    protected:

	virtual double animerLotS(long nbIterationByLot);

    private:

	void work();

	void fpsProcess(double* tabFpsi , Chrono* ptrChronoGlobal);

	void sizingLot();
	double sizingLotS();
	void sizingLotDetails();

	bool isBandewidthAvailable();

	static double gosBySeconde(double go , double durationS);
	static string unit(double go);
	static string arrondir(double go);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    protected:

	// Inputs
	Runnable_I* ptrRunnable;

	// Tools
	long nbIterationByLot;

    private:

	// Inputs

	string title;
	bool isVerboseEnable;
	double nbSecondesProcessingMax;

	// Inputs/Outputs Optional
	double inputGO;
	double outputGO;

	// Tools
	long nbLot;
	long nbIterationTotal;

	double durationLotEstimerS;

	// Output
	double durationS;

	long fpsGlobal;
	long fpsMean;
	long fpsMedian;
	long fpsMin;
	long fpsMax;
	long fpsStd;

	double durationLotStdS;
	double durationLotMaxS;
	double durationLotMinS;
	double durationLotMedianS;
	double durationLotGlobalS;
	double durationLotMeanS;

	double pourcentageErreur;

	//double gosInputGlobal;
	//double gosInputMean;
	double gosInputMedian;

	//double gosOutputGlobal;
	//double gosOutputMean;
	double gosOutputMedian;

//	double durationGlobalIteration1_MS;
//	double durationMeanIteration1_MS;
//	double durationMedianIteration1_MS;

	// ok en c++17, mais pas c++14
//	inline static const double PRECISION_CHRONO_S = 0.001;
//	inline static const double SEUIL_S = 0.02; // 20x plus grand

	static const double PRECISION_CHRONO_S;
	static const double SEUIL_S;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

