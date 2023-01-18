

/*   convert csr format to custom block-diagonal-fmt */
__global__
void csr_to_bd(csr *csrmat,
                c_float *bdmat,
                unsigned bdim,
                unsigned mdim){
    
    c_int col, row_start, row_end, bd_block_row, bd_block_col, bd_row, bd_col;
    c_float val;
    unsigned row, iter;

    
    for(row = GATO_BLOCK_ID*GATO_NUM_THREADS+GATO_THREAD_ID; row < csrmat->m; row +=GATO_NUM_THREADS*GATO_NUM_BLOCKS){    

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


__global__
void bd_to_csr(c_float *bdmat,
                csr *csrmat,
                unsigned bdim,
                unsigned mdim){

    c_int row, col, csr_row_offset, basic_col_offset, bd_block_row, bd_block_col, bd_col, bd_row, bd_row_len;
    unsigned iter, bd_offset;

    // each thread takes one row
    for(row = GATO_BLOCK_ID*GATO_NUM_THREADS+GATO_THREAD_ID; row < csrmat->m; row += GATO_NUM_THREADS*GATO_NUM_BLOCKS){

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

    }
}