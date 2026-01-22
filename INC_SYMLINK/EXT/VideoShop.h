#pragma once

#include <string>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class VideoShop
    {

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	static std::string neilpryde(int* ptrW , int* ptrH,int* ptrNbImage);
	static std::string autoroute(int* ptrW , int* ptrH,int* ptrNbImage);

	static std::string matrix22(int* ptrW , int* ptrH,int* ptrNbImage);
	static std::string matrix21(int* ptrW , int* ptrH,int* ptrNbImage);
	static std::string matrix12(int* ptrW , int* ptrH,int* ptrNbImage);

	static std::string contrast(int* ptrW , int* ptrH,int* ptrNbImage);
	static std::string dilatation(int* ptrW , int* ptrH,int* ptrNbImage);
	static std::string errosion(int* ptrW , int* ptrH,int* ptrNbImage);

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

