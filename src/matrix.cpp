

#include "matrix.h"

//column first indexing
int sub_to_ind(int *matSize, int r, int c, int matIndexing)
{
	//checks on the input coordinates
	if(r<0 || c <0 || r>=matSize[0] || c>=matSize[1])
	{
		std::cout << "Error, indices are incorrect for this matrix."<< std::endl;
		std::cout << "r : "<< r << ", c : " << c << std::endl;
		return(-1);
	}
	
	if (matIndexing==ROW_FIRST_MAT_INDEXING) 	//row first matrix indexing
	{
		return( r*matSize[1]+c);
	}
	else if(matIndexing==COL_FIRST_MAT_INDEXING)  //col first matrix indexing
	{
		return( c*matSize[0]+r);
	}
	else
	{
		std::cout << "Error in sub_to_ind, the matrix indexing type is not recognised." << std::endl;
		return(-1);
	}
}

int sub_to_ind(int m, int n, int r, int c, int matIndexing)
{
	//checks on the input coordinates
	if(r<0 || c <0 || r>=m || c>=n)
	{
		std::cout << "Error, indices are incorrect for this matrix."<< std::endl;
		std::cout << "r : "<< r << ", c : " << c << std::endl;
		std::cout << "m : "<<m << ", n : " << n << std::endl;
		return(-1);
	}
	
	if (matIndexing==ROW_FIRST_MAT_INDEXING) 	//row first matrix indexing
	{
		return( r*n+c);
	}
	else if(matIndexing==COL_FIRST_MAT_INDEXING)  //col first matrix indexing
	{
		return( c*m+r);
	}
	else
	{
		std::cout << "Error in sub_to_ind, the matrix indexing type is not recognised." << std::endl;
		std::cout << "Indexing" << matIndexing << std::endl;
		return(-1);
	}
}

//empty constructor
template <typename T>
matrix<T>::matrix( void )
{
	matValues = NULL;
	m=0;
	n=0;
	matIndexing=DEFAULT_MAT_INDEXING;
}

//constructors
template <typename T>
matrix<T>::matrix( int height, int width, int indexing)
{
	matValues = new T[height*width];
	m=height;
	n=width;
	matIndexing=indexing;
}

template <typename T>
matrix<T>::matrix( matrix* matIn )
{
	m = matIn->get_nrows();
	n = matIn->get_ncols();
	matIndexing = matIn->get_indexing();
	matValues = new T[m*n];
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = matIn->get_value(i,j);
}

template <typename T>
matrix<T>::~matrix(void)
{
	if (matValues != NULL)
		delete[] matValues;

}

//memory allocation
template <typename T>
int matrix<T>::allocate_memory(int height, int width)
{
	if (matValues != NULL)
	{
		std::cout << "Error, this is not an empty matrix object !" << std::endl;
		return(-1);
	}
	matValues = new T[height*width];
	m = height;
	n = width;
	return(0);
}

//accessors
template <typename T>
T matrix<T>::get_value(int r, int c)
{
	return( matValues[sub_to_ind(m, n, r, c,matIndexing)] );
}

template <typename T>
T* matrix<T>::get_ptr()
{
	return( matValues );
}

template <typename T>
void matrix<T>::set_ptr(T *ptrIn)
{
	if (matValues != NULL)
		delete matValues;
	matValues = ptrIn;
}

template <typename T>
T matrix<T>::operator()(int r, int c)
{
	return( matValues[sub_to_ind(m, n, r, c,matIndexing)] );
}

template <typename T>
void matrix<T>::normalise_matrix()
{
	T maxValue = this->max();
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			T valTemp = matValues[sub_to_ind(m, n, i, j,matIndexing)];
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = (valTemp)/(maxValue);
		}
}

template <typename T>
void matrix<T>::set_value(int r, int c, T valIn)
{
	matValues[sub_to_ind(m, n, r, c,matIndexing)] = valIn;
}

template <typename T>
void matrix<T>::set_to_zero()
{
	/*for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = (T)0.0;
		}*/
	std::memset(matValues, 0, m*n*sizeof(T));
}

template <typename T>
int matrix<T>::get_nrows()
{
	return(m);
}

template <typename T>
int matrix<T>::get_ncols()
{
	return(n);
}

template <typename T>
int matrix<T>::get_indexing()
{
	return(matIndexing);
}


//resizing/deletion operations
template <typename T>
void matrix<T>::delete_rows(int a, int b)
{
	if ( a > b || a<0 || b>=m)
	{
		std::cout << "Error in deleting the rows of the matrix, incorrect indices." << std::endl;
	}
	int nRowsToDelete = b-a+1;
	//create intermediate table to store the values
	int mNew = ( m-nRowsToDelete);
	int nNew = ( n);
	T *tabTemp = new T[ mNew * nNew];

	//copy first part of matrix
	int currRowIndex = 0;
	for (int i=0; i<a; i++)
	{
		for (int j=0; j<n; j++)
		{
			tabTemp[sub_to_ind(mNew,nNew,i,j,matIndexing)] = matValues[sub_to_ind(m,n,i,j,matIndexing)];
		}
		currRowIndex++;

	}
	
	//copy second part of matrix
	for (int i=(b+1); i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			tabTemp[sub_to_ind(mNew,nNew,currRowIndex,j,matIndexing)] = matValues[sub_to_ind(m,n,i,j,matIndexing)];
		}
		currRowIndex++;
	}
	//delete old table
	delete matValues;
	matValues = new T[ mNew * nNew];
	for (int i=0; i<mNew; i++)
		for (int j=0; j<nNew; j++)
		{
			T valTemp = tabTemp[sub_to_ind(mNew,nNew,i,j,matIndexing)];
			matValues[sub_to_ind(mNew,nNew,i,j,matIndexing)] = valTemp;
			//std::cout<< "ind 2" << sub_to_ind(mNew,nNew,i,j,matIndexing) << std::endl;
		}
	delete tabTemp;
	//reset the size of the matrix
	m =mNew;
	n = nNew;
}

template <typename T>
void matrix<T>::copy_mat_values(matrix *matIn)
{
	if (matIn->get_nrows() != m || matIn->get_ncols() != n)
		std::cout << "Error, the matrices' dimensions are not equal." << std::endl;
	
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m,n,i,j,matIndexing)] = matValues[sub_to_ind(m,n,i,j,matIndexing)];
		}
}

template <typename T>
T matrix<T>::max()
{
	T maxVal = std::numeric_limits<T>::min();
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			if ( matValues[sub_to_ind(m, n, i, j,matIndexing)] > maxVal )
				maxVal = matValues[sub_to_ind(m, n, i, j,matIndexing)];
		}
	return(maxVal);
}

template <typename T>
T matrix<T>::min()
{
	T minVal = std::numeric_limits<T>::max(); 
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			if ( matValues[sub_to_ind(m, n, i, j,matIndexing)] < minVal )
				minVal = matValues[sub_to_ind(m, n, i, j,matIndexing)];
		}
	return(minVal);
}

template <typename T>
T matrix<T>::max_col(int c)
{
	T maxVal = std::numeric_limits<T>::min();
	for (int i=0; i<m; i++)
	{
		if ( matValues[sub_to_ind(m, n, i, c,matIndexing)] > maxVal )
			maxVal = matValues[sub_to_ind(m, n, i, c,matIndexing)];
	}
	return(maxVal);
}

template <typename T>
T matrix<T>::min_col(int c)
{
	T minVal = std::numeric_limits<T>::max(); 
	for (int i=0; i<m; i++)
	{
		if ( matValues[sub_to_ind(m, n, i, c,matIndexing)] < minVal )
			minVal = matValues[sub_to_ind(m, n, i, c,matIndexing)];
	}
	return(minVal);
}

template <typename T>
void matrix<T>::add(T a)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = matValues[sub_to_ind(m, n, i, j,matIndexing)]+a;
		}
}

template <typename T>
void matrix<T>::subtract(T a)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = matValues[sub_to_ind(m, n, i, j,matIndexing)]-a;
		}
}

template <typename T>
void matrix<T>::multiply(T a)
{
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = matValues[sub_to_ind(m, n, i, j,matIndexing)]*a;
		}
}

template <typename T>
void matrix<T>::divide(T a)
{
	if(a == 0)
		std::cout<< "Error division by 0" << std::endl;
	for (int i=0; i<m; i++)
		for (int j=0; j<n; j++)
		{
			matValues[sub_to_ind(m, n, i, j,matIndexing)] = matValues[sub_to_ind(m, n, i, j,matIndexing)]/a;
		}
}

/******************************************/
/*********   INPUT/OUTPUT FUNCTIONS    ****/
/******************************************/

template <typename T>
void matrix<T>::print_matrix()
{
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			std::cout << matValues[sub_to_ind(m,n,i,j,matIndexing)] << " ";
		}
		std::cout << std::endl;
	}

}

template <typename T>
void matrix<T>::write_matrix(std::string outputFileName)
{

	std::ofstream outputFile;
	outputFile.open (outputFileName.c_str());

	//first, write the dimensions of the matrix
	outputFile << m << " " << n << "\n";
	for (int i=0; i<m; i++)
	{
		for (int j=0; j<n; j++)
		{
			if (j==(n-1))
				outputFile << matValues[sub_to_ind(m, n, i, j,matIndexing)];
			else
				outputFile << matValues[sub_to_ind(m, n, i, j,matIndexing)] << " ";
		}
		outputFile << "\n";
	}
	outputFile.close();
}

template <typename T>
void matrix<T>::write_matrix_col(int colNb)
{

	std::ofstream outputFile;
	outputFile.open ("output_file.txt");
	for (int i=0; i<m; i++)
	{
		outputFile << matValues[sub_to_ind(m, n, i, colNb,matIndexing)] << "\n";
	}
	outputFile.close();
}

/*template <typename T>
void matrix<T>::read_matrix(std::string inputFileName, int additionalSizeY,int additionalSizeX)
{

	std::ifstream inputFile;
	inputFile.open (inputFileName.c_str());

	//first, read the dimensions of the matrix, and allocate the space
	char bufTemp[100];
	int mIn,nIn;
	inputFile >> bufTemp;
	mIn = atoi(bufTemp);
	inputFile >> bufTemp;
	nIn = atoi(bufTemp);
	
	std::cout << "Matrix size : m : " << mIn << ", n : " << nIn << std::endl;
	//allocate the space for the matrix
	this->allocate_memory(mIn + additionalSizeY, nIn + additionalSizeX);
	
	//read the matrix
	int matIndX = 0;
	int matIndY = 0;
	while (true)
	{
		inputFile >> bufTemp;
		if( inputFile.eof() )
			break;
		matValues[sub_to_ind(m,n,matIndY,matIndX)] = (T)atof(bufTemp);
		matIndX++;
		if (matIndX>=nIn)
		{
			matIndX = 0;
			matIndY++;
		}
	}
	inputFile.close();
}*/

//instanciations of matrix types
template class matrix<double>;
template class matrix<float>;
template class matrix<int>;
template class matrix<bool>;


