#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <immintrin.h>
#include "mmio.h"
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
    int *csrRowPtrA;
    int *csrColIdxA;
    double *csrValA;
    int i, j, k;

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

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    double *csrValA_tmp    = (double *)malloc(nnzA_mtx_report * sizeof(double));

    // NOTE: when reading in doubles, ANSI C requires the use of the "l"  
    //   specifier as in "%lg", "%lf", "%le", otherwise errors will occur 
    //  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            

    for (i = 0; i < nnzA_mtx_report; i++)
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

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for ( i = 0; i < nnzA_mtx_report; i++ )
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)_mm_malloc((m+1) * sizeof(int), 64);
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)_mm_malloc(nnzA * sizeof(int), 64);
    csrValA    = (double *)_mm_malloc(nnzA * sizeof(double), 64);

    if (isSymmetric)
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    // srand(time(NULL));

    // set csrValA to 1, easy for checking floating-point results
    for (i = 0; i < nnzA; i++)
    {
        //csrValA[i] = rand() % 10;
    }

    printf("row: %d col: %d nnz: %d\n", m, n, nnzA);

    double *rhs = (double *)malloc(sizeof(double) * n);
    double *sol = (double *)malloc(sizeof(double) * n);
    double *x0 = (double *)malloc(sizeof(double) * n);
    double *x_pre = (double *)malloc(sizeof(double) * n);
    double *x_cur = (double *)malloc(sizeof(double) * n);
    double *h = (double *)malloc(sizeof(double) * n);
    double *r = (double *)malloc(sizeof(double) * n); 
    double *s = (double *)malloc(sizeof(double) * n);  
    double *t = (double *)malloc(sizeof(double) * n);    
    double *r0 = (double *)malloc(sizeof(double) * n);
    double *r_pre = (double *)malloc(sizeof(double) * n);
    double *r_cur = (double *)malloc(sizeof(double) * n);
    double *v_pre = (double *)malloc(sizeof(double) * n);
    double *v_cur = (double *)malloc(sizeof(double) * n);
    double *p_pre = (double *)malloc(sizeof(double) * n);
    double *p_cur = (double *)malloc(sizeof(double) * n);
    memset(v_pre, 0, sizeof(double) * n);
    memset(p_pre, 0, sizeof(double) * n);
    memset(v_cur, 0, sizeof(double) * n);
    memset(p_cur, 0, sizeof(double) * n);
    int max_ite = 1000;
    double alpha = 1;
    double beta;
    double *omega = (double *)malloc(sizeof(double) * n); 
    double *rho = (double *)malloc(sizeof(double) * n);
    omega[0] = 1;
    rho[0] = 1;

    for ( i = 0; i < n; i++ ) 
    {
        rhs[i] = 1.0;
        r[i] = 1.0;
    }
    double sum = 0; 
    for ( i = 0; i < n; i++ )
    {
        for ( j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++ )
        {
            sum += csrValA[j] * 1;
        }
        r0[i] = rhs[i] - sum;
        //printf("r0: %lf\n", r0[i]);
        r_pre[i] = r0[i];
        r_cur[i] = r0[i];
        x0[i] = 1;
        x_pre[i] = 1;
        x_cur[i] = 1;
        sum = 0;
    }

    double temp = 0;
    double error = 0;
    int flag = 0;
    double sum1 = 0;
    double sum2 = 0;
    for ( i = 1; i < max_ite; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
            r_pre[j] = r_cur[j];
            rho[i] += r0[j] * r_pre[j];
        }
        beta = ((double)rho[i]/(double)rho[i-1])*((double)alpha/(double)omega[i-1]);
        for ( j = 0; j < n; j++ )
        {
            p_pre[j] = p_cur[j];
            r_pre[j] = r_cur[j];
            v_pre[j] = v_cur[j];
            p_cur[j] = r_pre[j] + beta*(p_pre[j] - omega[i-1]*v_pre[j]);
        }
        sum = 0;
        for ( j = 0; j < n; j++ )
        {
            for ( k = csrRowPtrA[j]; k < csrRowPtrA[j+1]; k++ )
            {
                sum += csrValA[k] * p_cur[csrColIdxA[k]];
            }
            v_cur[j] = sum;
            sum = 0;
        }
        temp = 0;
        for ( j = 0; j < n; j++ )
        {
            temp += r0[j] * v_cur[j];
        }
        alpha = (double)rho[i]/(double)(temp);
        for ( j = 0; j < n; j++ )
        {
            x_pre[j] = x_cur[j];
            h[j] = x_pre[j] + alpha * p_cur[j];
        }
        error = 0;
        for ( j = 0; j < n; j++ )
        {
            for ( k = csrRowPtrA[j]; k < csrRowPtrA[j+1]; k++ )
            {
                error += csrValA[k] * h[csrColIdxA[k]];
            }
            if ( rhs[j] - error > 0.00001 )
            {
                flag = 1;
                break;
            }
            error = 0;
        }
        if ( flag == 0 ) 
        {
            for ( j = 0; j < n; j++ )
            {
                sol[j] = h[j];
                printf("1: sol[%d] = %lf h = %lf\n", j, sol[j], h[j]);
            }
            break;
        }
        flag = 0;
        for ( j = 0; j < n; j++ )
        {
            r_pre[j] = r_cur[j];
            s[j] = r_pre[j] - alpha*v_cur[j];
        }
        sum = 0;
        for ( j = 0; j < n; j++ )
        {
            for ( k = csrRowPtrA[j]; k < csrRowPtrA[j+1]; k++ )
            {
                sum += csrValA[k] * s[csrColIdxA[k]];
            }
            t[j] = sum;
            sum = 0;
        }
        sum1 = 0;
        sum2 = 0;
        for ( j = 0; j < n; j++ )
        {
            sum1 += t[j] * s[j];
        }
        for ( j = 0; j < n; j++ )
        {
            sum2 = t[j] * t[j];
        }
        omega[i] = (double)sum1/(double)sum2;
        for ( j = 0; j < n; j++ )
        {
            x_cur[j] = h[j] + omega[i]*s[j];
        }
        error = 0;
        for ( j = 0; j < n; j++ )
        {
            for ( k = csrRowPtrA[j]; k < csrRowPtrA[j+1]; k++ )
            {
                error += csrValA[k] * x_cur[csrColIdxA[k]];
            }
            if ( rhs[j] - error > 0.00001 )
            {
                flag = 1;
                break;
            }
            error = 0;
        }
        if ( flag == 0 ) 
        {
            for ( j = 0; j < n; j++ )
            {
                sol[j] = x_cur[j];
                printf("2: sol[%d] = %lf i = %d\n", j, sol[j], i);
            }
            break;
        }
        flag = 0;
        for ( j = 0; j < n; j++ )
        {
            r_cur[j] = s[j] - omega[i]*t[j];
        }
    }

    for ( i = 0; i < n; i++ )
    {
       // printf("sol[%d] = %lf\n", i, sol[i]);
    }

    return 0;
}