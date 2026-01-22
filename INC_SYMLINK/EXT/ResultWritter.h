#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include "Folders.h"

using std::string;
using std::ofstream;
using std::cout;
using std::endl;
using std::cerr;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class ResultWritter
    {

    public:

	/**
	 * <pre>
	 * Write the Matrice in the texte file filename.
	 *
	 * Sample :
	 *
	 * 	nxm = 10x5, separator= ','
	 *
	 * 	12,15,78,0,6
	 * 	10,1,6,7,-5
	 * 	...
	 * 	1,2,3,4,5
	 *
	 * Constrain :
	 *
	 * 	T must implements operator<< !
	 * </pre>
	 */
	template<typename T>
	static bool save(T* ptrMatrice , unsigned int n , unsigned int m , string filename , string folder,string separator = ",");

    };

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

template<typename T>
bool ResultWritter::save(T* ptrMatrice , unsigned int n , unsigned int m , string filename , string folder, string separator)
    {
    Folders::mkdirP(folder);

    string fileNameFull=folder+"/"+filename;

    ofstream filestream;
    filestream.open(fileNameFull.c_str());

    if (filestream.is_open())
	{
	for (int i = 0; i < n; i++)
	    {
	    for (int j = 0; j < m; j++)
		{
		int s = (i * m) + j; //Thread2D => Indice1D row-major
		filestream << ptrMatrice[s];

		if (j < m - 1) // Pour ne pas ajouter le separateur en fin de ligne
		    {
		    filestream << separator;
		    }
		}

	    filestream << "\n"; // Saut de ligne
	    }

	filestream.close();
	return true;
	}

    return false;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
