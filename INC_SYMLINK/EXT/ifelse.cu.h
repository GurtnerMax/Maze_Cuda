#pragma once

//#include "cudas.h"

/**
 * Supression thread divergence:
 *
 * 	if (t)
 * 		return a;
 * 	else
 * 		return b;
 *
 * qui est identique a a thread divergence de l'operateur ternaire
 *
 * 	return t?a:b
 */
template < typename T >
static __device__ T ifelse(bool t , T a , T b)
    {
    return (1 - t) * b + t * a;
    }

