#pragma once

#include <curand_kernel.h>

#include "Thread2D.cu.h"


/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/**
 * <pre>
 *  Utilisation:
 *  	(1) Fabriquer sur le device les generateurs parallels de nomber aleatoire : setup_kernel_rand
 *  	(2) Passer au kernel "user" ces generateurs : ptrDevTabGeneratorThread
 *  	(3) Dans le kernel "user"
 *  		 curandState generatorThread = ptrDevTabGeneratorThread[tid]; //Optimisation
 *  		 xAlea = curand_uniform(&generatorThread);
 * </pre>
 */
__global__ void setup_kernel_rand(curandState* ptrDevTabGeneratorThread , int deviceId)
    {
    const int TID = Thread2D::tid();

    // Customisation du generator: Proposition (au lecteur de faire mieux)
    // Contrainte : Doit etre diff�rent d'un GPU � l'autre
    int deltaSeed = deviceId * INT_MAX;
    int deltaSequence = deviceId * 100;
    int deltaOffset = deviceId * 100;

    int seed = 1234 + deltaSeed;    // deviceId+tid;
    int sequenceNumber = TID + deltaSequence;    // + tid;
    int offset = deltaOffset;

    //Each thread gets same seed , a different sequence number , no offset
    curand_init(seed, sequenceNumber, offset, &ptrDevTabGeneratorThread[TID]);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
