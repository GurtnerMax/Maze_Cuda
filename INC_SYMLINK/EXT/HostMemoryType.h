#pragma once

/*----------------------------------------------------------------------*\
 |*			Class	 					*|
 \*---------------------------------------------------------------------*/

/**
 * WRITE_COMBINED = PRIORITY DEVICE
 * PORTABLE = MULTI GPU
 */
enum HostMemoryType
    {
    DEFAULT,
    MULTIGPU,
    MAPPED,
    PRIORITYDEVICE,
    MAPPED_MULTIGPU,
    MAPPED_PRIORITYDEVICE,
    MAPPED_PRIORITYDEVICE_MULTIGPU
    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
