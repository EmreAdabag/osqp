#ifndef PCG_UTILS
#define PCG_UTILS

#include <iostream>
#include <iomanip>

#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

namespace gato{

/*******************************************************************************
*                              loady things                                    *
*******************************************************************************/

template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
    unsigned ind;
    for(ind=GATO_THREAD_ID; ind < size_Ts; ind+=GATO_THREADS_PER_BLOCK){
        dst[ind] = src[ind];
    }
}

// src is a B_DIM X B_DIM column-major matrix
// dst is a diagonal-format, block-tri, column_major, M_DIM*B_DIM X M_DIM*B_DIM matrix where col = 0, 1, 2 indicates left-diag, main-diag, right-diag
// offset might be needed but I don't think so
// multiplier multiplies src before storing
template <typename T, unsigned B_DIM, unsigned M_DIM>
__device__
void storeblock(T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1){
    
    if(col>2)
        return;
    // EMRE fix 0
    for(unsigned ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
        
        unsigned col_offset = (col*B_DIM+ind/B_DIM) * B_DIM*M_DIM;
        unsigned row_offset = BLOCKNO*B_DIM + ind%B_DIM;

        dst[col_offset + row_offset] = src[ind] * multiplier;
    }
}

template <typename T, unsigned B_DIM, unsigned M_DIM>
__device__
void storeblock_funnyformat(T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1){
    
    unsigned block_row_offset, block_col_offset, ind;

    if(col>2)
        return;


    block_row_offset = BLOCKNO * (3 * B_DIM * B_DIM);
    block_col_offset = col*B_DIM*B_DIM;


    if(multiplier==1){

        gato_memcpy<T>(
            dst+block_row_offset+block_col_offset,
            src,
            B_DIM*B_DIM
        );

    }
    else{
        
        for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}

template <typename T, unsigned B_DIM, unsigned M_DIM>
__device__
void loadblock_funnyformat(T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false){
    
    // EMRE assert this
    if(bcol > 2 || brow > M_DIM-1)
        return;
    

    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * B_DIM * B_DIM);
    block_col_offset = bcol*B_DIM*B_DIM;

    if(!transpose){

        gato_memcpy<T>(
            dst,
            src+block_row_offset+block_col_offset,
            B_DIM*B_DIM
        );

    }
    else{

        unsigned ind, transpose_col, transpose_row;

        for(ind=GATO_THREAD_ID; ind<B_DIM*B_DIM; ind+=GATO_THREADS_PER_BLOCK){
            transpose_col = ind%B_DIM * B_DIM;
            transpose_row = ind/B_DIM;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
        }
    }
}

template <typename T, unsigned BLOCK_DIM>
__device__
void loadBlockTriDiagonal_offDiagonal(T *s_var, T *d_var_b, cgrps::thread_block block, cgrps::grid_group grid){
    // Need to load b also now
    for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
        s_var[ind + BLOCK_DIM] = *(d_var_b + ind); 
    }
    // if first block just want b and b+1 (and already have b)
    if(GATO_LEAD_BLOCK){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind + 2*BLOCK_DIM] = *(d_var_b + BLOCK_DIM + ind); // just b+1
        }

    }
    // if last block just want b-1 and b (and already have b)
    else if (GATO_LAST_BLOCK){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind] = *(d_var_b - BLOCK_DIM + ind); // just b-1
        }

    }
    // else want b-1 and b and b+1 (and already have b)
    else{
        for (unsigned ind = GATO_THREAD_ID; ind < 2*BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            T *dst, *src;
            if (ind < BLOCK_DIM){dst = s_var + ind;       	  src = d_var_b - BLOCK_DIM + ind;} // b-1
            else		  		{dst = s_var + BLOCK_DIM + ind; src = d_var_b + ind;} // b+1
            *dst = *src;
        }
    }
}

template <typename T, unsigned BLOCK_DIM>
__device__ 
void matVecMultBlockTriDiagonal(T *s_dst, T *s_mat, T *s_vec, cgrps::thread_block block, cgrps::grid_group grid){
    // First or Last block only 2 mults (var and either var+1 or var-1)
    if(GATO_LEAD_BLOCK){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM*BLOCK_DIM + BLOCK_DIM * c + r] * s_vec[c + BLOCK_DIM]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (GATO_LAST_BLOCK){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c]; // var and var-1
            }
            s_dst[r] = val;
        }
    }
    // else 3 mults
    else{
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

template <typename T, unsigned BLOCK_DIM, unsigned N>
__device__
void loadBlockTriDiagonal_offDiagonal_fixed(T *s_var, T *d_var_b, unsigned block_row, cgrps::thread_block block, cgrps::grid_group grid){
    // Need to load b also now
    for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
        s_var[ind + BLOCK_DIM] = *(d_var_b + ind); 
    }
    // if first block just want b and b+1 (and already have b)
    if(block_row == 0){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind + 2*BLOCK_DIM] = *(d_var_b + BLOCK_DIM + ind); // just b+1
        }

    }
    // if last block just want b-1 and b (and already have b)
    else if (block_row == N){
        for (unsigned ind = GATO_THREAD_ID; ind < BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            s_var[ind] = *(d_var_b - BLOCK_DIM + ind); // just b-1
        }

    }
    // else want b-1 and b and b+1 (and already have b)
    else{
        for (unsigned ind = GATO_THREAD_ID; ind < 2*BLOCK_DIM; ind += GATO_THREADS_PER_BLOCK){
            T *dst, *src;
            if (ind < BLOCK_DIM){dst = s_var + ind;       	  src = d_var_b - BLOCK_DIM + ind;} // b-1
            else		  		{dst = s_var + BLOCK_DIM + ind; src = d_var_b + ind;} // b+1
            *dst = *src;
        }
    }
}

template <typename T, unsigned BLOCK_DIM, unsigned N>
__device__ 
void matVecMultBlockTriDiagonal_fixed(T *s_dst, T *s_mat, T *s_vec, unsigned block_row, cgrps::thread_block block, cgrps::grid_group grid){
    // First or Last block only 2 mults (var and either var+1 or var-1)
    if(block_row == 0 ){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM*BLOCK_DIM + BLOCK_DIM * c + r] * s_vec[c + BLOCK_DIM]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (block_row == N){
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c]; // var and var-1
            }
            s_dst[r] = val;
        }
    }
    // else 3 mults
    else{
        for (unsigned r = GATO_THREAD_ID; r < BLOCK_DIM; r += GATO_THREADS_PER_BLOCK){
            T val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void reducePlus(T *dstTemp, cgrps::thread_block block){
    unsigned size_left = VEC_SIZE;
    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = GATO_THREAD_ID; ind < size_left; ind += GATO_THREADS_PER_BLOCK){
            dstTemp[ind] += dstTemp[ind + size_left];
        }	
        // add the odd size adjust if needed
        if (GATO_LEAD_THREAD && odd_flag){dstTemp[0] += dstTemp[2*size_left];}
        // sync and repeat
        block.sync();
    }
    // when we get really small sum up what is left
    if (GATO_LEAD_THREAD){
        for(unsigned ind = 1; ind < size_left; ind++){dstTemp[0] += dstTemp[ind];}
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void dotProd(T *dstTemp, T *vec1, T *vec2, cgrps::thread_block block){
    // first compute temp sums across all threads
    for (unsigned ind = GATO_THREAD_ID; ind < VEC_SIZE; ind += GATO_THREADS_PER_BLOCK){
        dstTemp[ind] = vec1[ind] * vec2[ind];
    }
    block.sync();
    // then reduce
    reducePlus<T,VEC_SIZE>(dstTemp,block);
}

/*******************************************************************************
*                              printy things                                   *
*******************************************************************************/



template <typename T, unsigned M, unsigned N>
__host__ __device__
void print_block(T *A){
    for(unsigned i=0; i<M; i++){
        for(unsigned j=0; j<N; j++){printf("%.4f  ",A[i + M*j]);}
        printf("\n\n");
    }
} 

template <typename T, unsigned B_DIM, unsigned M_DIM>
__host__ __device__
void print_raw_funny_format_matrix(T *A){

    unsigned row_size, block_size;
    unsigned block_row, block_col, row, col;
    unsigned i, j;

    row_size = 3 * B_DIM * B_DIM;
    block_size = B_DIM * B_DIM;

    for(i=0; i < B_DIM * M_DIM; i++){
        for(j=0; j < B_DIM * 3; j++){

            block_row = i / B_DIM; 
            block_col = j / B_DIM; 
            row = i % B_DIM;
            col = j % B_DIM;
            
            printf("%.4f  ",A[ block_row*row_size + block_col*block_size + col*B_DIM + row ]);
        }
        printf("\n\n");
    }
} 


// __host__ __device__
// template <typename T, unsigned STATE_SIZE>
// void print_raw_shared_matrix(T *A){

//     unsigned row_size, col_size;
//     unsigned i, j;

//     row_size = 3 * STATE_SIZE;
//     col_size = STATE_SIZE;

//     for(int i=0;i<row_size;i++){
//         for(int j=0; j<col_size;j++){
//             printf("%0.4f  ", A[j*STATE_SIZE*STATE_SIZE + i]);
//         }
//         printf("\n");
//     }
// } 


// __host__ __device__
// template <typename T, unsigned ROW_SIZE>
// void print_raw_shared_vector(T *A){
//     unsigned i;

//     for(int i=0;i<ROW_SIZE;i++){
//         printf("%0.4f  ", A[i]); 
//     }
//     printf("\n");
// }



template<typename T, unsigned block_dim_Ts, unsigned matrix_dim_blocks>
__host__
void print_funny_format_matrix(T *A){

    const unsigned bd = block_dim_Ts;
    const unsigned md = matrix_dim_blocks;
    const unsigned blocksize = bd * bd;
    const unsigned rowsize = 3 * blocksize;
    
    unsigned row, col;
    int blockrow, blockcol;

    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    
    for(row = 0; row < bd * md; row++){
        
        blockrow = row / bd;

        if(blockrow==0){
            for(col=0; col < bd * md; col++){
                
                blockcol = col / bd;
                if(blockcol < 2){ std::cout << A[ (col+bd)*bd + row ] << "\t"; }
                else{ std::cout << static_cast<T>(0) << "\t"; }
            }
        }
        else{
            for(col=0; col < bd * md; col++){
                
                blockcol = col / bd;
                if(blockcol < blockrow-1 || blockcol > blockrow+1){ 
                    std::cout << static_cast<T>(0) << "\t"; 
                }
                else{ 
                    std::cout << A[ rowsize*blockrow + (blockcol-blockrow+1)*blocksize + (col%bd)*bd + row%bd ] << "\t"; 
                }
                
            }
        }
        
        std::cout << "\n\n";
    }
}




/*******************************************************************************
*                            matrix inversion                                  *
*******************************************************************************/
    

// load identity in so memory is [A | I]
template <typename T, unsigned DIM>
__device__ __forceinline__
void loadIdentity(T *A){
    for (unsigned ind = GATO_THREAD_ID; ind < DIM*DIM; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c;
        r = ind % DIM; 
        c = ind / DIM;
        A[ind] = static_cast<T>(r == c);
    }
}

// load identity in so memory is [V | I]
template <typename T, unsigned DIMA, unsigned DIMB>
__device__ __forceinline__
void loadIdentity(T *A, T *B){
    for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c, indAdj; T *V;
        if (ind < DIMA*DIMA){
            indAdj = ind;
            r = indAdj % DIMA; c = indAdj/DIMA; V = A;
        }
        else {
            indAdj = ind - DIMA*DIMA;
            r = indAdj % DIMB; c = indAdj/DIMB; V = B;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}


// load identity in so memory is [V | I]
template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC>
__device__ __forceinline__
void loadIdentity(T *A, T *B, T *C){
    for (unsigned ind = GATO_THREAD_ID; ind < DIMA*DIMA+DIMB*DIMB+DIMC*DIMC; ind += GATO_THREADS_PER_BLOCK){
        unsigned r, c, indAdj; T *V;
        if (ind < DIMA*DIMA){
            indAdj = ind;
            r = indAdj % DIMA; c = indAdj/DIMA; V = A;
        }
        else if (ind < DIMA*DIMA+DIMB*DIMB){
            indAdj = ind - DIMA*DIMA;
            r = indAdj % DIMB; c = indAdj/DIMB; V = B;
        }
        else{
            indAdj = ind - DIMA*DIMA - DIMB*DIMB;
            r = indAdj % DIMC; c = indAdj/DIMC; V = C;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}


template <typename T, unsigned DIM>
__device__
void invertMatrix(T *A, T *s_temp){ 
// we are going to guassian elimination walking down the matrix (assuming no leading 0s)
// we therefore use the columns in order as the pivot column for each pivot we need to rescale 
// that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
// of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
// pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    for (unsigned pivRC = 0; pivRC < DIM; pivRC++){
        unsigned pivColOffset = pivRC*DIM;
        // save the pivot and pivot column and row
        T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
        for (unsigned ind = GATO_THREAD_ID; ind < 2*DIM+1; ind++){
            unsigned AInd;
            if (ind < DIM){AInd = ind + pivColOffset;}
            else{AInd = pivRC + pivColOffset + (ind-DIM)*DIM;}
            s_temp[ind] = A[AInd];
        }
        __syncthreads(); //----------------------
        // make the pivot update
        for (unsigned ind = GATO_THREAD_ID; ind < DIM*(DIM+1); ind += GATO_THREADS_PER_BLOCK){
            unsigned row = ind % DIM; unsigned col = ind / DIM; unsigned colOffset = ind - row;
            // s_temp = orpcvs|prvOld
            if (row == pivRC){A[row + pivColOffset + colOffset] *= pvInv;}
            else{A[row + pivColOffset + colOffset] -= s_temp[row]*pvInv*s_temp[DIM+col];}
        }
    __syncthreads(); //----------------------
    }
}


template <typename T, unsigned DIMA, unsigned DIMB, unsigned MAX_DIM>
__device__
void invertMatrix(T *A, T *B, T *s_temp){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
        bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB;
        unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
            if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
            if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
            if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
        }
        __syncthreads(); //----------------------
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < DIMA*(DIMA+1)){
                unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
            }
            if (BActive && ind < DIMB*(DIMB+1)){
                unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
            }
        }
        __syncthreads(); //----------------------
    }
}

// invert A,B,C assume memory for all is [V | VInv] where both are DIMxDIM and continguous
// relies on s_temp of size [2*DIMA + 2*DIMB + 2*DIMC + 3]
template <typename T, unsigned DIMA, unsigned DIMB, unsigned DIMC, unsigned MAX_DIM>
__device__
void invertMatrix(T *A, T *B, T *C, T *s_temp){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    T *s_memA = s_temp; T *s_memB = &s_memA[2*DIMA+1]; T *s_memC = &s_memB[2*DIMB+1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++){
        bool AActive = pivRC < DIMA; bool BActive = pivRC < DIMB; bool CActive = pivRC < DIMC;
        unsigned pivColOffsetA = pivRC*DIMA; unsigned pivColOffsetB = pivRC*DIMB; unsigned pivColOffsetC = pivRC*DIMC;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM; ind++){
            if (AActive && ind < DIMA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < DIMB){s_memB[ind] = B[ind + pivColOffsetB];}
            if (CActive && ind < DIMC){s_memC[ind] = C[ind + pivColOffsetC];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM+1; ind++){
            if (AActive && ind < DIMA+1){s_memA[ind + DIMA] = A[ind*DIMA + pivRC + pivColOffsetA];}
            if (BActive && ind < DIMB+1){s_memB[ind + DIMB] = B[ind*DIMB + pivRC + pivColOffsetB];}
            if (CActive && ind < DIMC+1){s_memC[ind + DIMC] = C[ind*DIMC + pivRC + pivColOffsetC];}
        }
        __syncthreads(); //----------------------
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < MAX_DIM*(MAX_DIM+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < DIMA*(DIMA+1)){
                unsigned row = ind % DIMA; unsigned col = ind / DIMA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[DIMA+col];}
            }
            if (BActive && ind < DIMB*(DIMB+1)){
                unsigned row = ind % DIMB; unsigned col = ind / DIMB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[DIMB+col];}
            }
            if (CActive && ind < DIMC*(DIMC+1)){
                unsigned row = ind % DIMC; unsigned col = ind / DIMC;
                if (row == pivRC){C[pivColOffsetC + ind] /= s_memC[pivRC];}
                else{C[pivColOffsetC + ind] -= s_memC[row]/s_memC[pivRC]*s_memC[DIMC+col];}
            }
        }
        __syncthreads(); //----------------------
    }
}



/*******************************************************************************
*                             Integrator things                                *
*******************************************************************************/

/*
template<typename T>
__host__ __device__ 
T angleWrap(T input){
    const T pi = static_cast<T>(3.14159);
    if(input > pi){input = -(input - pi);}
    if(input < -pi){input = -(input + pi);}
    return input;
}


template <typename T, unsigned STATE_SIZE, unsigned CONTROL_SIZE, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator_error(T *s_err, T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt){
    T new_qkp1; T new_qdkp1;
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE/2; ind += GATO_THREADS_PER_BLOCK){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            new_qkp1 = s_q[ind] + dt*s_qd[ind];
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            new_qdkp1 = s_qd[ind] + dt*s_qdd[ind];
            new_qkp1 = s_q[ind] + dt*new_qdkp1;
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){ printf("ANGLE_WRAP!\n");
            new_qkp1 = angleWrap(new_qkp1);
        }

        // then computre error
        s_err[ind] = abs(s_qkp1[ind] - new_qkp1);
        s_err[ind + STATE_SIZE/2] = abs(s_qdkp1[ind] - new_qdkp1);
        // printf("err[%f] with new qkp1[%f] vs orig[%f] and new qdkp1[%f] vs orig[%f] with qk[%f] qdk[%f] qddk[%f] and dt[%f]\n",s_err[ind],new_qkp1,s_qkp1[ind],new_qdkp1,s_qdkp1[ind],s_q[ind],s_qd[ind],s_qdd[ind],dt);
    }
}

template <typename T, unsigned STATE_SIZE, unsigned CONTROL_SIZE, unsigned INTEGRATOR_TYPE = 0>
__device__
void exec_integrator_gradient(T *s_Ak, T *s_Bk, T *s_dqdd, T dt){
        
    // and finally A and B
    if (INTEGRATOR_TYPE == 0){
        // then apply the euler rule -- xkp1 = xk + dt*dxk thus AB = [I_{state},0_{control}] + dt*dxd
        // where dxd = [ 0, I, 0; dqdd/dq, dqdd/dqd, dqdd/du]
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE*(STATE_SIZE + CONTROL_SIZE); ind += GATO_THREADS_PER_BLOCK){
            int c = ind / STATE_SIZE; int r = ind % STATE_SIZE;
            T *dst = (c < STATE_SIZE)? &s_Ak[ind] : &s_Bk[ind - STATE_SIZE*STATE_SIZE]; // dst
            T val = (r == c) * static_cast<T>(1); // first term (non-branching)
            val += (r < STATE_SIZE/2 && r == c - STATE_SIZE/2) * dt; // first dxd term (non-branching)
            val += (r >= STATE_SIZE/2) * dt * s_dqdd[c*STATE_SIZE/2 + r - STATE_SIZE/2]; // second dxd term (non-branching)
            *dst = val;
        }
    }
    else if (INTEGRATOR_TYPE == 1){
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1 = qk + dt*qdk + dt^2*qddk
        // dxkp1 = [Ix | 0u ] + dt*[[0q, Iqd, 0u] + dt*dqdd
        //                                             dqdd]
        // Ak = I + dt * [[0,I] + dt*dqdd/dx; dqdd/dx]
        // Bk = [dt*dqdd/du; dqdd/du]
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            int c = ind / STATE_SIZE; int r = ind % STATE_SIZE; int rdqdd = r % (STATE_SIZE/2);
            T dtVal = static_cast<T>((r == rdqdd)*dt + (r != rdqdd));
            s_Ak[ind] = static_cast<T>((r == c) + dt*(r == c - STATE_SIZE/2)) +
                        dt * s_dqdd[c*STATE_SIZE/2 + rdqdd] * dtVal;
            if(c < CONTROL_SIZE){
                s_Bk[ind] = dt * s_dqdd[STATE_SIZE*STATE_SIZE/2 + c*STATE_SIZE/2 + rdqdd] * dtVal;
            }
        }
    }
    else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}
}


template <typename T, unsigned STATE_SIZE, unsigned CONTROL_SIZE, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__device__ 
void exec_integrator(T *s_qkp1, T *s_qdkp1, T *s_q, T *s_qd, T *s_qdd, T dt){
    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE/2; ind += GATO_THREADS_PER_BLOCK){
        // euler xk = xk + dt *dxk
        if (INTEGRATOR_TYPE == 0){
            s_qkp1[ind] = s_q[ind] + dt*s_qd[ind];
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
        }
        // semi-inplicit euler
        // qdkp1 = qdk + dt*qddk
        // qkp1 = qk  + dt*qdkp1
        else if (INTEGRATOR_TYPE == 1){
            s_qdkp1[ind] = s_qd[ind] + dt*s_qdd[ind];
            s_qkp1[ind] = s_q[ind] + dt*s_qdkp1[ind];
        }
        else{printf("Integrator [%d] not defined. Currently support [0: Euler and 1: Semi-Implicit Euler]",INTEGRATOR_TYPE);}

        // wrap angles if needed
        if(ANGLE_WRAP){
            s_qkp1[ind] = angleWrap(s_qkp1[ind]);
        }
    }
}

// s_temp of size STATE_SIZE/2*(STATE_SIZE + CONTROL_SIZE + 1) + DYNAMICS_TEMP
template <typename T, unsigned STATE_SIZE, unsigned CONTROL_SIZE, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false, bool COMPUTE_INTEGRATOR_ERROR = false>
__device__ __forceinline__
void integratorAndGradient(T *s_xux, T *s_Ak, T *s_Bk, T *s_xnew_err, T *s_temp, void *d_dynMem_const, T dt){
    // first compute qdd and dqdd
    T *s_qdd = s_temp; 	
    T *s_dqdd = s_temp + STATE_SIZE/2;	
    T *s_extra_temp = s_dqdd + STATE_SIZE/2*(STATE_SIZE+CONTROL_SIZE);
    T *s_q = s_xux; 	
    T *s_qd = s_q + STATE_SIZE/2; 		
    T *s_u = s_qd + STATE_SIZE/2;
    gato_plant::forwardDynamicsAndGradient<T>(s_dqdd, s_qdd, s_q, s_qd, s_u, s_extra_temp, d_dynMem_const);
    __syncthreads(); //----------------------
    // first compute xnew or error
    if (COMPUTE_INTEGRATOR_ERROR){
        exec_integrator_error<T,STATE_SIZE,CONTROL_SIZE,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, s_xux, &s_xux[STATE_SIZE/2], s_q, s_qd, s_qdd, dt);
    }
    else{
        exec_integrator<T,STATE_SIZE,CONTROL_SIZE,INTEGRATOR_TYPE,ANGLE_WRAP>(s_xnew_err, &s_xnew_err[STATE_SIZE/2], s_q, s_qd, s_qdd, dt);
    }
    
    // then compute gradient
    exec_integrator_gradient<T,STATE_SIZE,CONTROL_SIZE,INTEGRATOR_TYPE>(s_Ak, s_Bk, s_dqdd, dt);
}



*/

/*******************************************************************************
*                             matrix operations                                *
*******************************************************************************/


template <typename T, unsigned MAT_ROWS, unsigned MAT_COLS>
__device__
void mat_vec_prod(T *mat, T *vec, T *out){
    
    for(unsigned row=0; row<MAT_ROWS; row++){
        T res = static_cast<T>(0);
        for (unsigned col = 0; col < MAT_COLS; col++){
            res += mat[row + col*MAT_ROWS] * vec[col];
        }
        out[row] = res;
    }
}


template <typename T, unsigned MAT_A_ROWS, unsigned MAT_A_COLS, unsigned MAT_B_ROWS, unsigned MAT_B_COLS, bool transposeB>
__device__
void mat_mat_prod(T *mat_A, T *mat_B, T *out){

    if(!transposeB){
        if(MAT_A_COLS!=MAT_B_ROWS){
            printf("this should be an assert but matmat was fed wrong");
            return;
        }

        for(unsigned row=0; row<MAT_A_ROWS; row++){

            for (unsigned col = 0; col < MAT_B_COLS; col++){
                T res = static_cast<T>(0);

                for(unsigned ind=0; ind<MAT_A_COLS; ind++){
                    res += mat_A[ind*MAT_A_ROWS + row] * mat_B[col*MAT_B_ROWS+ind];
                }

                out[col*MAT_A_ROWS+row] = res;
            }
        }
    }
    else{                       // transpose matrix B

        if(MAT_A_COLS!=MAT_B_COLS){
            printf("this should be an assert but matmat was fed wrong");
            return;
        }

        for(unsigned row=0; row<MAT_A_ROWS; row++){

            for (unsigned rowB= 0; rowB<MAT_B_ROWS; rowB++){
                T res = static_cast<T>(0);

                for(unsigned ind=0; ind<MAT_A_COLS; ind++){
                    res += mat_A[ind*MAT_A_ROWS + row] * mat_B[ind*MAT_B_ROWS+rowB];
                }

                out[rowB*MAT_A_ROWS+row] = res;
            }
        }

    }
}


/*******************************************************************************
*                              format conversion                               *
*******************************************************************************/

/*   convert csr format to custom block-diagonal-fmt */
__device__
void csr_to_bd(csr *csrmat,
                c_float *bdmat,
                unsigned bdim,
                unsigned mdim){
    
    c_int col, row_start, row_end, bd_block_row, bd_block_col, bd_row, bd_col;
    c_float val;
    unsigned row, iter;

    
    for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < csrmat->m; row +=GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){    

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        for(iter=row_start; iter<row_end; iter++){
            col = csrmat->col_ind[iter];
            val = csrmat->val[iter];

            bd_block_row = ( row / bdim );                     // block row
            bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
            bd_col = col % bdim;
            bd_row = row % bdim;

            bdmat[ bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row] = val;
        }
    }

}

__device__
void csr_to_std(csr *csrmat,
                c_float *stdmat){
    
    c_int col, row_start, row_end;
    c_float val;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    for(row; row < csrmat->m; row +=step){    

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        for(iter=row_start; iter<row_end; iter++){
            col = csrmat->col_ind[iter];
            val = csrmat->val[iter];

            stdmat[col*csrmat->m + row] = val;
        }
    }
}


__device__
void bd_to_csr(c_float *bdmat,
                csr *csrmat,
                unsigned bdim,
                unsigned mdim){

    c_int row, col, csr_row_offset, basic_col_offset, bd_block_row, bd_block_col, bd_col, bd_row, bd_row_len;
    unsigned iter, bd_offset;

    // each thread takes one row
    for(row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID; row < csrmat->m; row += GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS){

        bd_block_row = row/bdim;

        // row_len
        if(bd_block_row==0 || bd_block_row==mdim-1){
            bd_row_len = 2*bdim;
        }
        else{
            bd_row_len = 3*bdim;
        }

        // set row_ptr
        if(bd_block_row==0){                        // first block
            csr_row_offset = row*bd_row_len;
            basic_col_offset = 0;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
            if(row==0){
                csrmat->row_ptr[row] = 0;
            }
        }
        else if(bd_block_row==mdim-1){              // last block
            csr_row_offset = 2*bdim*bdim+(mdim-2)*3*bdim*bdim+(row%bdim)*bd_row_len;
            basic_col_offset = (bd_block_row-1)*bdim;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
        }
        else{
            csr_row_offset = 2*bdim*bdim+(row-bdim)*bd_row_len;
            basic_col_offset = (bd_block_row-1)*bdim;

            csrmat->row_ptr[row+1] = csr_row_offset+bd_row_len;
        }

        for(iter=0; iter<bd_row_len; iter++){

            col = basic_col_offset+iter;
            bd_block_row = ( row / bdim );                     // block row
            bd_block_col = ( col / bdim ) + 1 - bd_block_row;  // block col
            bd_col = col % bdim;
            bd_row = row % bdim;

            bd_offset = bd_block_row*3*bdim*bdim + bd_block_col*bdim*bdim + bd_col*bdim + bd_row;
            
            csrmat->col_ind[csr_row_offset+iter] = col;
            csrmat->val[csr_row_offset+iter] = bdmat[bd_offset];
        }

        if(row==csrmat->m-1){
            csrmat->nnz = STATE_SIZE*STATE_SIZE*3*KNOT_POINTS;
        }

    }
}


__device__
void csr_to_custom_G(csr *csrmat,
                     c_float *d_G){


    /*
    out size   (STATES_SQ+CONTROLS_SQ)*KNOT_POINTS-CONTROLS_SQ

    output must be initialized to zeroes
    */
    
    c_int row_start, row_end, in_set_row, set_offset, in_set_col;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    for(; row < csrmat->m; row +=step){    

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        in_set_row = row % (STATE_SIZE+CONTROL_SIZE);
        set_offset = (row / (STATE_SIZE + CONTROL_SIZE)) * (STATES_SQ + CONTROLS_SQ);

        for(iter=row_start; iter<row_end; iter++){

            in_set_col = csrmat->col_ind[iter] % (STATE_SIZE+CONTROL_SIZE);

            if( in_set_col < STATE_SIZE){
                d_G[set_offset + in_set_col * STATE_SIZE +in_set_row] = csrmat->val[iter];
            }
            else{
                d_G[set_offset + STATES_SQ + (in_set_col - STATE_SIZE) * CONTROL_SIZE + (in_set_row - STATE_SIZE)] = csrmat->val[iter];
            }
        }
    }
}


__device__
void csr_to_custom_C(csr *csrmat,
                     c_float *d_C){

    /*
    out size   (STATES_SQ+STATES_P_CONTROLS)*(KNOT_POINTS-1)*sizeof(c_float)

    output must be initialized to zeroes
    */

    c_int col, row_start, row_end, block_row;
    unsigned row, step, iter;
    
    row = GATO_BLOCK_ID*GATO_THREADS_PER_BLOCK+GATO_THREAD_ID;
    step = GATO_THREADS_PER_BLOCK*GATO_NUM_BLOCKS;
    
    // step through rows
    for(; row < csrmat->m; row +=step){

        if(row < STATE_SIZE){continue;}

        row_start = csrmat->row_ptr[row];
        row_end = csrmat->row_ptr[row+1];

        block_row = (row / STATE_SIZE)-1;

        for(iter=row_start; iter<row_end; iter++){
            
            col = csrmat->col_ind[iter];
            if((col/(STATE_SIZE+CONTROL_SIZE))>block_row){continue;}

            d_C[ block_row*(STATES_SQ+STATES_P_CONTROLS)
                            + (col % (STATE_SIZE+CONTROL_SIZE)) * STATE_SIZE
                            + (row % (STATE_SIZE)) ] = csrmat->val[iter];

        }
    }
}

__global__
void gato_convert_kkt_format(cudapcg_solver *s, c_float *d_G, c_float *d_C, c_float *d_g){
    
    // convert C to custom dense format
    csr_to_custom_C(s->A, d_C);

    // convert G to custom dense format
    csr_to_custom_G(s->P, d_G);

    // copy g into d_g
    gato_memcpy<c_float>(d_g, s->d_rhs, ((STATE_SIZE+CONTROL_SIZE)*KNOT_POINTS-CONTROL_SIZE)*sizeof(c_float));

    // TODO: mirror C, G if upper triangular only

}

}
#endif      /* #ifndef PCG_UTILS */
