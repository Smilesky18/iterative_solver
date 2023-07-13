#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mmio.h"
#include "papi.h"
#define COLOR_NONE "\033[0m" //表示清除前面设置的格式
#define RED "\033[1;31;40m" //40表示背景色为黑色, 1 表示高亮
#define BLUE "\033[1;34;40m"
#define GREEN "\033[1;32;40m"
#define YELLOW "\033[1;33;40m"
#define MICRO_IN_SEC 1000000.00
double microtime()
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv,&tz);
    
    return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

int main(int argc, char ** argv)
{
    int m, n, nnzA;
    int *cscColPtrA;
    int *cscRowIdxA;
    double *cscValA;

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    printf("--------------%s--------------------\n", filename);

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf ("Could not process Matrix Market banner.\n" );
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1;} //cout << "type = Pattern" << endl;
    if ( mm_is_real ( matcode) )     { isReal = 1;} //cout << "type = real" << endl;
    if ( mm_is_complex( matcode ) ) { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1;} //cout << "type = integer" << endl

    // find out size of sparse matrix .... 
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        printf("symmetric = true\n");
    }
    else
    {
        printf("symmetric = false\n");
    }

    int *cscColPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(cscColPtrA_counter, 0, (m+1) * sizeof(int));

    int *cscRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *cscColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    double *cscValA_tmp    = (double *)malloc(nnzA_mtx_report * sizeof(double));

    // NOTE: when reading in doubles, ANSI C requires the use of the "l"  
    //   specifier as in "%lg", "%lf", "%le", otherwise errors will occur 
    //  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;

        if (isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isComplex)
        {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        cscColPtrA_counter[idxi]++;
        cscRowIdxA_tmp[i] = idxi;
        cscColIdxA_tmp[i] = idxj;
        cscValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (cscRowIdxA_tmp[i] != cscColIdxA_tmp[i])
                cscColPtrA_counter[cscRowIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = cscColPtrA_counter[0];
    cscColPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = cscColPtrA_counter[i];
        cscColPtrA_counter[i] = old_val + cscColPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = cscColPtrA_counter[m];
    cscColPtrA = (int *)_mm_malloc((m+1) * sizeof(int), 64);
    memcpy(cscColPtrA, cscColPtrA_counter, (m+1) * sizeof(int));
    memset(cscColPtrA_counter, 0, (m+1) * sizeof(int));

    cscRowIdxA = (int *)_mm_malloc(nnzA * sizeof(int), 64);
    cscValA    = (double *)_mm_malloc(nnzA * sizeof(double), 64);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (cscRowIdxA_tmp[i] != cscColIdxA_tmp[i])
            {
                int offset = cscColPtrA[cscColIdxA_tmp[i]] + cscColPtrA_counter[cscColIdxA_tmp[i]];
                cscRowIdxA[offset] = cscRowIdxA_tmp[i];
                cscValA[offset] = cscValA_tmp[i];
                cscColPtrA_counter[cscColIdxA_tmp[i]]++;

                offset = cscColPtrA[cscRowIdxA_tmp[i]] + cscColPtrA_counter[cscRowIdxA_tmp[i]];
                cscRowIdxA[offset] = cscColIdxA_tmp[i];
                cscValA[offset] = cscValA_tmp[i];
                cscColPtrA_counter[cscRowIdxA_tmp[i]]++;
            }
            else
            {
                int offset = cscColPtrA[cscColIdxA_tmp[i]] + cscColPtrA_counter[cscColIdxA_tmp[i]];
                cscRowIdxA[offset] = cscRowIdxA_tmp[i];
                cscValA[offset] = cscValA_tmp[i];
                cscColPtrA_counter[cscColIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = cscColPtrA[cscColIdxA_tmp[i]] + cscColPtrA_counter[cscColIdxA_tmp[i]];
            cscRowIdxA[offset] = cscRowIdxA_tmp[i];
            cscValA[offset] = cscValA_tmp[i];
            cscColPtrA_counter[cscColIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(cscRowIdxA_tmp);
    free(cscValA_tmp);
    free(cscColIdxA_tmp);
    free(csccolPtrA_counter);

    printf("row: %d col: %d nnz: %d\n", m, n, nnzA);

    double *rhs = (double *)malloc(sizeof(double) * n);
    double *sol = (double *)malloc(sizeof(double) * n);
    double *r = (double *)malloc(sizeof(double) * n);  
    double *r0 = (double *)malloc(sizeof(double) * n);
    double *r_pre = (double *)malloc(sizeof(double) * n);
    double *r_cur = (double *)malloc(sizeof(double) * n);
    double *v_pre = (double *)malloc(sizeof(double) * n);
    double *v_cur = (double *)malloc(sizeof(double) * n);
    double *p_pre = (double *)malloc(sizeof(double) * n);
    double *p_cur = (double *)malloc(sizeof(double) * n);
    memset(v_pre, 0, sizeof(double) * n);
    memset(p_pre, 0, sizeof(double) * n);
    int max_ite = 100;
    double alpha = 1;
    double *omega = (double *)malloc(sizeof(double) * n); 
    double *rho = (double *)malloc(sizeof(double) * n);

    for ( int i = 0; i < n; i++ ) 
    {
        rhs[i] = 1.0;
        r[i] = 1.0;
    } 
    for ( int i = 0; i < n; i++ )
    {
        
    }

    for ( int i = 1; i < max_ite; i++ )
    {

    }

    return 0;
}