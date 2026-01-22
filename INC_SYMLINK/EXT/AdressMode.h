#pragma once

/*----------------------------------------------------------------------*\
 |*			Enum	 					*|
 \*---------------------------------------------------------------------*/

/**
 * adressMode:
 * 	- cudaAddressModeBorder
 * 	- cudaAddressModeMirror
 * 	- cudaAddressModeWrap(modulo) DEFAULT
 * 	- cudaAddressModeClamp (nearest)
 */
enum AdressMode
    {
    DEFAUT_ADRESS_MODE, // WRAP
    BORDER, //
    MIRROR, //
    WRAP, //  DEFAULT modulo
    CLAMP // nearest
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
