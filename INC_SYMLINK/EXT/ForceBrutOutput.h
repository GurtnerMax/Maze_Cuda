#pragma once

#include <string>
#include <ostream>

#include "GridFps.h"
#include "GridMaillage.h"

using std::string;
using std::ostream;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

class ForceBrutOutput
    {
	/*--------------------------------------*\
	 |*		Constructor		*|
	 \*-------------------------------------*/

    public:

	ForceBrutOutput(GridFps* tabGridFps , GridMaillage* ptrGridMaillage , string title);

	virtual ~ForceBrutOutput();

	/*--------------------------------------*\
	 |*		Methodes		*|
	 \*-------------------------------------*/

    public:

	/**
	 * folder example : "."
	 */
	void save(string file , string folder);

	/**
	 * folder example : "."
	 */
	void save(string folder);

	void print(ostream& stream);

	friend ostream& operator <<(ostream& stream , ForceBrutOutput& forceBrutOutput);

	/*--------------------------------------*\
	 |*		Get			*|
	 \*-------------------------------------*/

	/**
	 * <pre>
	 *  Dans le cas ou la grille est generer automatiquement par le choix des iterateur, on a:
	 *
	 * -----> db
	 * |
	 * |
	 * dg
	 *</pre>
	 */
	long* getTabFps() const;

	GridFps* getTabGridFps() const;

	GridFps* getTabGridFpsSorted() const;

	GridFps* getTabGridFpsMax(int* n) const;

	Grid* getTabGrid() const;

	int getSize() const;

	int getN() const;

	int getM() const;

	std::string getTitle();

	std::string getFile();

	std::string getFolder();

    private:

	void process();

	static GridFps* clone(GridFps* tab , int n);

	static void print(GridFps* tab , int n);

	static void print(long* tab , int n);

	/*--------------------------------------*\
	 |*		Attributs		*|
	 \*-------------------------------------*/

    private:

	// Inputs
	GridMaillage* ptrGridMaillage;
	string title;
	int size;
	int n;
	int m;
	GridFps* tabGridFps;

	// Outputs
	GridFps* tabGridFpsSorted;
	GridFps* tabGridFpsMax;
	long* tabFps;
	int nbFpsMax;
	std::string file;
	std::string folder;

    };

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
