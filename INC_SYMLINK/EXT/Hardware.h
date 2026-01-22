#pragma once

#include <string>
#include <assert.h>
#include <cuda_runtime.h>

#include "Grid.h"
#include "GpuFamily.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/**
 * idDevice in [0,nbDevice-1]
 *
 * dim3.x
 * dim3.y
 * dim3.z
 *
 * sm=1.2 major=1 minor=2
 */
class Hardware
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/*--------------*\
	|* 	set      *|
	 \*-------------*/

	static void setDevice(int deviceId);
	static void reset(int deviceId);

	/**
	 * cudaDeviceReset causes the driver to clean up all state.
	 * While not mandatory in normal operation, it is good practice.
	 */
	static void reset();

	/*--------------*\
	|* 	get      *|
	 \*-------------*/

	static int getRuntimeVersion();
	static int getDriverVersion();

	static int getDeviceCount();
	static int getDeviceId();

	/**
	 * total
	 */
	static int getCoreCount(int idDevice);

	/**
	 * total
	 */
	static int getCoreCount();

	/**
	 * by MP
	 */
	static int getCoreCountMP(int idDevice);

	/**
	 * by MP
	 */
	static int getCoreCountMP();

	static cudaDeviceProp getDeviceProp(int idDevice);
	static cudaDeviceProp getDeviceProp();

	static std::string getNameSimple(int idDevice);
	static std::string getNameSimple();

	static std::string getName(int idDevice);
	static std::string getName();

	static int getWarpSize(int idDevice);
	static int getWarpSize();

	static int getMPCount();
	static int getMPCount(int idDevice);

	static int getCapacityMajor(int idDevice);
	static int getCapacityMajor();

	static int getCapacityMinor(int idDevice);
	static int getCapacityMinor();

	static int getAsyncEngineCount(int idDevice);
	static int getAsyncEngineCount();

	/*--------------*\
	|* 	max    *|
	 \*-------------*/

	static int getMaxThreadPerBlock(int idDevice);
	static int getMaxThreadPerBlock();

	static int getMaxThreadPerMP(int idDevice);
	static int getMaxThreadPerMP();

	static dim3 getMaxGridDim(int idDevice);
	static dim3 getMaxGridDim();

	static dim3 getMaxBlockDim(int idDevice);
	static dim3 getMaxBlockDim();

	/*--------------*\
	|* 	memory    *|
	 \*-------------*/

	/**
	 * Go
	 */
	static int getGM(int idDevice);

	/**
	 * GO
	 */
	static int getGM();

	/**
	 * ko
	 */
	static int getCM(int idDevice);

	/**
	 * ko
	 */
	static int getCM();

	/**
	 * ko
	 */
	static int getSM(int idDevice);

	/**
	 * ko
	 */
	static int getSM();

	/*--------------*\
	|* print       *|
	 \*-------------*/

	static void printAll();
	static void printAllSimple();

	static void print(int idDevice);
	static void print();

	static void printCurrent();

	/*--------------*\
	 |*	Is       *|
	 \*-------------*/

	static bool isUVAEnable(int idDevice);
	static bool isUVAEnable();

	static bool isHostMapMemoryEnable(int idDevice);
	static bool isHostMapMemoryEnable();

	static bool isECCEnable(int idDevice);
	static bool isECCEnable();

	static bool isAsyncEngine(int idDevice);
	static bool isAsyncEngine();

	/*--------------*\
	|*	Arch       *|
	 \*-------------*/

	static bool isCuda();//plante si pas cuda
	static bool isCuda(int idDevice);//plante si pas cuda

	static bool isFermi(int idDevice);
	static bool isFermi();

	static bool isKepler(int idDevice);
	static bool isKepler();

	static bool isMaxwell(int idDevice);
	static bool isMaxwell();

	static bool isPascal(int idDevice);
	static bool isPascal();

	static bool isVolta(int idDevice);
	static bool isVolta();

	static bool isTuring(int idDevice);
	static bool isTuring();

	static bool isAmpere(int idDevice);
	static bool isAmpere();

	// 2022
	static bool isHopper(int idDevice);
	static bool isHopper();

//	static bool isLovelace(int idDevice)
//	static bool isLovelace()

	static GpuFamily getFamily(int idDevice);
	static GpuFamily getFamily();

	/*--------------*\
	|*	load     *|
	 \*-------------*/

	/**
	 * Force load
	 */
	static void loadCudaDriver(int deviceID , bool isMapMemoryEnable = false);
	/**
	 * Force load
	 */
	static void loadCudaDriver(bool isMapMemoryEnable = false);
	/**
	 * Force load
	 */
	static void loadCudaDriverAll(bool isMapMemoryEnable = false);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

