#pragma once

#include "GM.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/**
 * see :
 *
 * 	Lock
 *
 * Note :
 *
 * 	Lock ne laisse aucune trace coter host, il s'instancie only coterdevice: Code moins invasif!
 * 	LockMixte laisse une trace coter host. Code plus invasif
 *
 * Use (host) :
 *
 *	LockMixte lock;
 *	kernel<<<dg,db>>>(...,lock,...); // pas besoin de memory managment MM
 *
 * Use (device):
 *
 * 	lock.lock();
 * 	doSomething();
 * 	lock.unlock();
 *
 */
class LockMixte // pas tester depuis longtemps!
    {

	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	LockMixte()
	    {
	    int state = 0;

	    GM::malloc<int>(&ptrDevMutexGM, sizeof(int));
	    GM::memcpyHToD<int>(ptrDevMutexGM, &state, sizeof(int));
	    }

	/**
	 * Observation:
	 *
	 * 	Comme Lock est passer par valeur au kernel,ie par copie,deux objets au total sont vivants, un sur le cpu et sun sur le gpu.
	 * 	Le destructeur sera donc appeler 2x,la premiere fois, lors de la fin du kernel sur le gpu, la seconde fois sur le cpu.
	 * 	Comme cudaFree peut etre appeler sur le cpu ou gpu, le code ci-dessous est valide, mais sans le HANDLE_ERROR qui n'a aucun sens
	 * 	(affichage d'un message d'error) sur le GPU et qui est donc enlever ici.
	 * 	Heureusement cudaFree peut etre appeler plusieurs fois de suite sans probleme,mais un seul appel suffirait!
	 *
	 * Attention:
	 *
	 * 	Sur le GPU, il ne faut jamais passer Lock par valeur, car sinon, la premiere instance detruite, detruit ptrDev_mutex !!
	 * 	et le mutex devient alors inutilisable!
	 */
	~LockMixte(void)
	    {
	    GM::free(ptrDevMutexGM);
	    }

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	__device__
	void lock(void)
	    {
	    while (atomicCAS((int*)ptrDevMutexGM, 0, 1) != 0);
	    }

	__device__
	void unlock(void)
	    {
	    //v1
	    atomicExch((int*)ptrDevMutexGM, 0); // tester ok

	    //v2
	    //*ptrDevMutexGM = 0; // tester ok

	    // new
	    __threadfence(); // ko pour v1 et v2 si pas la! (from cuda 10 et rtx 2080ti)
	    }

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	int* ptrDevMutexGM; // Espace adressage GPU, en global memory GM
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
