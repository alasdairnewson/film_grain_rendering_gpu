
#ifndef MATRIX_H
#define MATRIX_H

	#include <iostream>
	#include <fstream>
	#include <limits>
	#include <cstring>

	#define ROW_FIRST_MAT_INDEXING 0
	#define COL_FIRST_MAT_INDEXING 1
	#define DEFAULT_MAT_INDEXING 0  //0 for row first, 1 for column first
	//column first indexing
	int sub_to_ind(int *matSize, int r, int c, int matIndexing=DEFAULT_MAT_INDEXING);
	int sub_to_ind(int m, int n, int r, int c, int matIndexing=DEFAULT_MAT_INDEXING);

	template <typename T>
	class matrix {
		private:
		int m,n;
		T *matValues;
		int matIndexing;
		public:
			//constructors
			matrix( void );				//empty constructor
      		matrix( int height, int width, int indexing=DEFAULT_MAT_INDEXING );             // simple constructor
			matrix( matrix* matIn );	   //by copy
			//destructor
			~matrix();

			//memory allocation
			int allocate_memory(int height, int width);

			//accessors
			T get_value(int r, int c);
			T* get_ptr();
			T operator()(int r, int c);
			void set_value(int r, int c, T valIn);
			void set_ptr(T* ptrIn);
			
			int get_nrows();
			int get_ncols();
			int get_indexing();

			//deletion of rows/columns
			void delete_rows(int a, int b);

			void copy_mat_values(matrix *matIn);

			//misc functions
			void normalise_matrix();

			void set_to_zero();

			T max();
			T min();
			T max_col(int c);
			T min_col(int c);

			void add(T a);
			void subtract(T a);
			void multiply(T a);
			void divide(T a);

			//input/output functions
			void print_matrix();

			void write_matrix(std::string outputFileName);
			void write_matrix_col(int colNb);
			//void read_matrix(std::string inputFileName, int additionalSizeY=0,int additionalSizeX=0);
	};


#endif
