#include "gato_pcg_defines.h"

#include "gato_pcg_utils.h"

#include <cuda.h>
#include <cooperative_groups.h>

namespace cgrps = cooperative_groups;

namespace gato{

    /*******************************************************************************
    *                        private functions pcg init                            *
    *******************************************************************************/
        
    //  s_xux               xk | uk | xkp1
    //  s_xs                start state
    //  s_xg                goal state
    //  rho                 regularizer
    //  s_temp
    //  d_temp
    //  d_dynMem_const
    //  dt
    //  blockrow             matrix block number, NOT threadblock number (that is GATO_BLOCK_ID)


    __device__
    void gato_form_schur_jacobi_inner(c_float *d_G, c_float *d_C, c_float *d_g,c_float *d_S, c_float *d_Pinv, c_float *d_gamma, c_float *s_temp, unsigned blockrow){
        
        
        //  SPACE ALLOCATION IN SHARED MEM
        //  | phi_k | theta_k | thetaInv_k | gamma_k | block-specific...
        //     s^2      s^2         s^2         s
        c_float *s_phi_k = s_temp; 	                            	    // phi_k        states^2
        c_float *s_theta_k = s_phi_k + STATES_SQ; 			            // theta_k      states^2
        c_float *s_thetaInv_k = s_theta_k + STATES_SQ; 			        // thetaInv_k   states^2
        c_float *s_gamma_k = s_thetaInv_k + STATES_SQ;                       // gamma_k      states
        c_float *s_end_main = s_gamma_k + STATE_SIZE;                               

        if(blockrow==0){

            //  LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | . | Q_N_I | q_N | . | Q_0_I | q_0 | scatch
            //              s^2   s^2     s   s^2   s^2     s      ? 
        
            c_float *s_QN = s_end_main;
            c_float *s_QN_i = s_QN + STATE_SIZE * STATE_SIZE;
            c_float *s_qN = s_QN_i + STATE_SIZE * STATE_SIZE;
            c_float *s_Q0 = s_qN + STATE_SIZE;
            c_float *s_Q0_i = s_Q0 + STATE_SIZE * STATE_SIZE;
            c_float *s_q0 = s_Q0_i + STATE_SIZE * STATE_SIZE;
            c_float *s_end = s_q0 + STATE_SIZE;

            // scratch space
            c_float *s_integrator_error = s_end;					
            c_float *s_R_not_needed = s_integrator_error + STATE_SIZE;
            c_float *s_r_not_needed = s_R_not_needed + CONTROL_SIZE * CONTROL_SIZE;
            c_float *s_extra_temp = s_r_not_needed + CONTROL_SIZE * CONTROL_SIZE;

            __syncthreads();//----------------------------------------------------------------

            // TODO: make gato_memcpy
            gato_memcpy(s_Q0, d_G, STATES_SQ*sizeof(c_float));
            gato_memcpy(s_QN, d_G+(KNOT_POINTS-1)*(STATES_SQ+CONTROLS_SQ), STATES_SQ*sizeof(c_float));
            gato_memcpy(s_q0, d_g, STATE_SIZE*sizeof(c_float));
            gato_memcpy(s_qN, d_g+(KNOT_POINTS-1)*(STATE_SIZE+CONTROL_SIZE), STATE_SIZE*sizeof(c_float));

            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("Q0\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_Q0,STATE_SIZE);
            //     printf("q0\n");
            //     printMat<c_float,1,STATE_SIZE>(s_q0,1);
            //     printf("QN\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_QN,STATE_SIZE);
            //     printf("qN\n");
            //     printMat<c_float,1,STATE_SIZE>(s_qN,1);
            //     printf("start error\n");
            //     printMat<c_float,1,STATE_SIZE>(s_integrator_error,1);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------
            
            // SHARED MEMORY STATE
            // | Q_N | . | q_N | Q_0 | . | q_0 | scatch
            

            // save -Q_0 in PhiInv spot 00
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_Q0,                       // src     
                d_Pinv,                   // dst         
                1,                          // col
                blockrow,                    // blockrow
                -1                          //  multiplier
            );
            __syncthreads();//----------------------------------------------------------------


            // invert Q_N, Q_0
            loadIdentity<c_float, STATE_SIZE,STATE_SIZE>(s_Q0_i, s_QN_i);
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<c_float, STATE_SIZE,STATE_SIZE,STATE_SIZE>(s_Q0, s_QN, s_extra_temp);
            
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("Q0Inv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Q0_i,STATE_SIZE);
            //     printf("QNInv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_QN_i,STATE_SIZE);
            //     printf("theta\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_theta_k,STATE_SIZE);
            //     printf("thetaInv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_thetaInv_k,STATE_SIZE);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | . | Q_0_i | q_0 | scatch
            

            // compute gamma
            mat_vec_prod<c_float, STATE_SIZE, STATE_SIZE>(
                s_Q0_i,                                    
                s_q0,                                       
                s_gamma_k 
            );
            __syncthreads();//----------------------------------------------------------------
            

            // save -Q0_i in spot 00 in S
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_Q0_i,                         // src             
                d_S,                            // dst              
                1,                              // col   
                blockrow,                        // blockrow         
                -1                              //  multiplier   
            );
            __syncthreads();//----------------------------------------------------------------


            // compute Q0^{-1}q0
            mat_vec_prod<c_float, STATE_SIZE, STATE_SIZE>(
                s_Q0_i,
                s_q0,
                s_Q0
            );
            __syncthreads();//----------------------------------------------------------------


            // SHARED MEMORY STATE
            // | . | Q_N_i | q_N | Q0^{-1}q0 | Q_0_i | q_0 | scatch


            // save -Q0^{-1}q0 in spot 0 in gamma
            for(unsigned ind = GATO_THREAD_ID; ind < STATES_SQ; ind += GATO_THREADS_PER_BLOCK){
                d_gamma[ind] = -s_Q0[ind];
            }
            __syncthreads();//----------------------------------------------------------------

        }
        else{                       // blockrow!=LEAD_BLOCK


            const unsigned C_set_size = STATES_SQ+STATES_P_CONTROLS;
            const unsigned G_set_size = STATES_SQ+CONTROLS_SQ;

            //  NON-LEADING BLOCK GOAL SHARED MEMORY STATE
            //  ...gamma_k | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp
            //               s^2   s*c  s^2   s^2   s^2    s^2    s^2   s^2     s      s      s          s                <s^2?

            c_float *s_Ak = s_end_main; 								
            c_float *s_Bk = s_Ak +        STATES_SQ;
            c_float *s_Qk = s_Bk +        STATE_SIZE*CONTROL_SIZE; 	
            c_float *s_Qk_i = s_Qk +      STATES_SQ;	
            c_float *s_Qkp1 = s_Qk_i +    STATES_SQ;
            c_float *s_Qkp1_i = s_Qkp1 +  STATES_SQ;
            c_float *s_Rk = s_Qkp1_i +    STATES_SQ;
            c_float *s_Rk_i = s_Rk +      CONTROLS_SQ;
            c_float *s_qk = s_Rk_i +      CONTROLS_SQ; 	
            c_float *s_qkp1 = s_qk +      STATE_SIZE; 			
            c_float *s_rk = s_qkp1 +      STATE_SIZE;
            c_float *s_end = s_rk +       CONTROL_SIZE;
            
            // scratch
            c_float *s_integrator_error = s_end; 	
            c_float *s_extra_temp = s_integrator_error + STATE_SIZE;      // s_extra_temp size: max(2*STATES+1, 2*CONTROLS+1)
            

            // if(PRINT_THREAD){
            //     printf("xk\n");
            //     printMat<c_float1,STATE_SIZE>(s_xux,1);
            //     printf("uk\n");
            //     printMat<c_float1,CONTROL_SIZE>(&s_xux[STATE_SIZE],1);
            //     printf("xkp1\n");
            //     printMat<c_float1,STATE_SIZE>(&s_xux[STATE_SIZE+CONTROL_SIZE],1);
            //     printf("\n");
            // }

            __syncthreads();//----------------------------------------------------------------

            gato_memcpy(s_Ak,   d_C+      (blockrow-1)*C_set_size,                        STATES_SQ);
            gato_memcpy(s_Bk,   d_C+      (blockrow-1)*C_set_size+STATES_SQ,              STATES_P_CONTROLS);
            gato_memcpy(s_Qk,   d_G+      (blockrow-1)*G_set_size,                        STATES_SQ);
            gato_memcpy(s_Qkp1, d_G+    (blockrow*G_set_size),                          STATES_SQ);
            gato_memcpy(s_Rk,   d_G+      ((blockrow-1)*G_set_size+STATES_SQ),            CONTROLS_SQ);
            gato_memcpy(s_qk,   d_g+      (blockrow-1)*(STATES_S_CONTROLS),               STATE_SIZE);
            gato_memcpy(s_qkp1, d_g+    (blockrow)*(STATES_S_CONTROLS),                 STATE_SIZE);
            gato_memcpy(s_rk,   d_g+      ((blockrow-1)*(STATES_S_CONTROLS)+STATE_SIZE),  CONTROL_SIZE);

            __syncthreads();//----------------------------------------------------------------

        
            // if(PRINT_THREAD){
            //     printf("Ak\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Ak,STATE_SIZE);
            //     printf("Bk\n");
            //     printMat<c_floatSTATE_SIZE,CONTROL_SIZE>(s_Bk,STATE_SIZE);
            //     printf("Qk\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Qk,STATE_SIZE);
            //     printf("Rk\n");
            //     printMat<c_floatCONTROL_SIZE,CONTROL_SIZE>(s_Rk,CONTROL_SIZE);
            //     printf("qk\n");
            //     printMat<c_float1,STATE_SIZE>(s_qk,1);
            //     printf("rk\n");
            //     printMat<c_float1,CONTROL_SIZE>(s_rk,1);
            //     printf("Qkp1\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Qkp1,STATE_SIZE);
            //     printf("qkp1\n");
            //     printMat<c_float1,STATE_SIZE>(s_qkp1,1);
            //     printf("integrator error\n");
            //     printMat<c_float1,STATE_SIZE>(s_integrator_error,1);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------


            // Invert Q, Qp1, R
            loadIdentity<c_float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE>(
                s_Qk_i, 
                s_Qkp1_i, 
                s_Rk_i
            );
            __syncthreads();//----------------------------------------------------------------
            invertMatrix<c_float, STATE_SIZE,STATE_SIZE,CONTROL_SIZE,STATE_SIZE>(
                s_Qk, 
                s_Qkp1, 
                s_Rk, 
                s_extra_temp
            );
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("Qk\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Qk_i,STATE_SIZE);
            //     printf("RkInv\n");
            //     printMat<c_floatCONTROL_SIZE,CONTROL_SIZE>(s_Rk_i,CONTROL_SIZE);
            //     printf("Qkp1Inv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Qkp1_i,STATE_SIZE);
            //     printf("\n");
            // }
            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | A_k | B_k | . | Q_k_I | . | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp


            // Compute AQ^{-1} in phi
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, false>(
                s_Ak,
                s_Qk_i,
                s_phi_k
            );

            __syncthreads();//----------------------------------------------------------------

            // Compute BR^{-1} in Qkp1
            mat_mat_prod<c_float, STATE_SIZE, CONTROL_SIZE, CONTROL_SIZE, CONTROL_SIZE, false>(
                s_Bk,
                s_Rk_i,
                s_Qkp1
            );

            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | A_k | B_k | . | Q_k_I | BR^{-1} | Q_k+1_I | . | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp


            // compute Q_{k+1}^{-1}q_{k+1} - IntegratorError in gamma
            mat_vec_prod<c_float, STATE_SIZE, STATE_SIZE>(
                s_Qkp1_i,
                s_qkp1,
                s_gamma_k
            );
            for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
                s_gamma_k[i] -= s_integrator_error[i];
            }
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("AQinv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_phi_k,STATE_SIZE);
            //     printf("BRinv\n");
            //     printMat<c_floatSTATE_SIZE,CONTROL_SIZE>(s_Qkp1,STATE_SIZE);
            //     printf("Qpinv*qp\n");
            //     printMat<c_float1,STATE_SIZE>(s_gamma_k,1);
            //     printf("\n");
            // }

            __syncthreads();//----------------------------------------------------------------

            // compute AQ^{-1}q for gamma         temp storage in Rk
            mat_vec_prod<c_float, STATE_SIZE, STATE_SIZE>(
                s_phi_k,
                s_qk,
                s_Rk
            );

            __syncthreads();//----------------------------------------------------------------
            
            // compute BR^{-1}r for gamma           temp storage in Qk
            mat_vec_prod<c_float, STATE_SIZE, CONTROL_SIZE>(
                s_Qkp1,
                s_rk,
                s_Qk
            );

            __syncthreads();//----------------------------------------------------------------
            
            // gamma = yeah...
            for(unsigned i = GATO_THREAD_ID; i < STATE_SIZE; i += GATO_THREADS_PER_BLOCK){
                s_gamma_k[i] -= s_Rk[i] + s_Qk[i];
            }
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("BRinvB\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_BRB,STATE_SIZE);
            //     printf("s_BRr\n");
            //     printMat<c_float1,STATE_SIZE>(s_BRr,1);
            //     printf("AQinvA + Qkp1Inv\n");
            //     printMat<c_floatSTATE_SIZE,STATE_SIZE>(s_Qkp1Inv,STATE_SIZE);
            //     printf("Qpinv*qk - AQq\n");
            //     printMat<c_float,1,STATE_SIZE>(s_gamma_k,1);
            //     printf("AQinv\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_AQ,STATE_SIZE);
            //     printf("memory in phi_k\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_phi_k,STATE_SIZE);
            //     printf("\n");
            // }
            // __syncthreads();//----------------------------------------------------------------


            // SHARED MEMORY STATE
            // | A_k | B_k | BR^{-1}r | Q_k_I | BR^{-1} | Q_k+1_I | AQ^{-1}q | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp


            // compute AQ^{-1}A   +   Qkp1^{-1} for theta
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, true>(
                s_phi_k,
                s_Ak,
                s_theta_k
            );

            __syncthreads();//----------------------------------------------------------------

            for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
                s_theta_k[i] += s_Qkp1_i[i];
            }
            __syncthreads();//----------------------------------------------------------------


            // compute BR^{-1}BT for theta            temp storage in Rk
            mat_mat_prod<c_float, STATE_SIZE, CONTROL_SIZE, STATE_SIZE, CONTROL_SIZE, true>(
                s_Qkp1,
                s_Bk,
                s_Rk
            );

            __syncthreads();//----------------------------------------------------------------

            for(unsigned i = GATO_THREAD_ID; i < STATES_SQ; i += GATO_THREADS_PER_BLOCK){
                s_theta_k[i] += s_Rk[i];
            }
            __syncthreads();//----------------------------------------------------------------


            // if(PRINT_THREAD){
            //     printf("theta = -BRinvB - AQinvA - Qkp1Inv\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_theta_k,STATE_SIZE);
            //     printf("phi = AQ (before negation)\n");
            //     printMat<c_float,STATE_SIZE,STATE_SIZE>(s_phi_k,STATE_SIZE);
            //     printf("gamma = Qpinv*qk - AQq -BRr\n");
            //     printMat<c_float,1,STATE_SIZE>(s_gamma_k,1);
            // }
            __syncthreads();//----------------------------------------------------------------

            // SHARED MEMORY STATE
            // | A_k | B_k | BR^{-1}r | Q_k_I | BR^{-1} | Q_k+1_I | BR^{-1}BT | R_k_I | q_k | q_k+1 | r_k | integrator_error | extra_temp


            // save -phi_k into left off-diagonal of S, 
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_phi_k,                        // src             
                d_S,                            // dst             
                0,                              // col
                blockrow,                        // blockrow    
                1
            );
            __syncthreads();//----------------------------------------------------------------


            // save -theta_k main diagonal S
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_theta_k,                                               
                d_S,                                                 
                1,                                               
                blockrow,
                -1                                              
            );          
            __syncthreads();//----------------------------------------------------------------

#if BLOCK_J_PRECON || SS_PRECON
        // invert theta
        loadIdentity<c_float,STATE_SIZE>(s_thetaInv_k);
        __syncthreads();//----------------------------------------------------------------
        invertMatrix<c_float,STATE_SIZE>(s_theta_k, s_extra_temp);
        __syncthreads();//----------------------------------------------------------------


        // save thetaInv_k main diagonal PhiInv
        storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
            s_thetaInv_k, 
            d_Pinv,
            1,
            blockrow,
            -1
        );
#else /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */

        // save 1 / diagonal to PhiInv
        for(int i = GATO_THREAD_ID; i < STATE_SIZE; i+=GATO_THREADS_PER_BLOCK){
            d_Pinv[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]= 1 / d_S[blockrow*(3*STATES_SQ)+STATES_SQ+i*STATE_SIZE+i]; 
        }
#endif /* BLOCK_J_PRECONDITIONER || SS_PRECONDITIONER  */
        

        __syncthreads();//----------------------------------------------------------------

        // save gamma_k in gamma
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            unsigned offset = (blockrow)*STATE_SIZE + ind;
            d_gamma[offset] = s_gamma_k[ind]*-1;
        }

        __syncthreads();//----------------------------------------------------------------

        //invert phi_k
        loadIdentity<c_float,STATE_SIZE>(s_Ak);
        __syncthreads();//----------------------------------------------------------------
        mat_mat_prod<c_float,STATE_SIZE,STATE_SIZE,STATE_SIZE,STATE_SIZE, true>(s_Ak,s_phi_k,s_Bk);
        __syncthreads();//----------------------------------------------------------------

        // save phi_k_T into right off-diagonal of S,
        storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
            s_Bk,                        // src             
            d_S,                            // dst             
            2,                              // col
            blockrow-1                        // blockrow    
        );

        __syncthreads();//----------------------------------------------------------------
        }

    }


#if SS_PRECON
    __device__
    gato_form_ss_inner(c_float *s_temp, c_float *d_S, c_float *d_Pinv, c_float *d_gamma, unsigned blockrow){
        
        //  STATE OF DEVICE MEM
        //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal
        //  Phi:    -Q0 in spot 00, theta_invk main diagonal
        //  gamma:  -Q0_i*q0 spot 0, gammak


        // GOAL SPACE ALLOCATION IN SHARED MEM
        // s_temp  = | phi_k_T | phi_k | phi_kp1 | thetaInv_k | thetaInv_kp1 | thetaInv_km1 | PhiInv_R | PhiInv_L | scratch
        c_float *s_phi_k = s_temp;
        c_float *s_phi_kp1_T = s_phi_k + STATES_SQ;
        c_float *s_thetaInv_k = s_phi_kp1_T + STATES_SQ;
        c_float *s_thetaInv_km1 = s_thetaInv_k + STATES_SQ;
        c_float *s_thetaInv_kp1 = s_thetaInv_km1 + STATES_SQ;
        c_float *s_PhiInv_k_R = s_thetaInv_kp1 + STATES_SQ;
        c_float *s_PhiInv_k_L = s_PhiInv_k_R + STATES_SQ;
        c_float *s_scratch = s_PhiInv_k_L + STATES_SQ;

        // load phi_kp1_T
        if(blockrow!=LAST_BLOCK){
            loadblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                d_S,                // src
                s_phi_kp1_T,        // dst
                0,                  // block column (0, 1, or 2)
                blockrow+1,          // block row
                true                // transpose
            );
        }
        
        __syncthreads();//----------------------------------------------------------------

        // load phi_k
        if(blockrow!=LEAD_BLOCK){
            loadblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                d_S,
                s_phi_k,
                0,
                blockrow
            );
        }
        
        __syncthreads();//----------------------------------------------------------------


        // load thetaInv_k
        loadblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
            d_Pinv,
            s_thetaInv_k,
            1,
            blockrow
        );

        __syncthreads();//----------------------------------------------------------------

        // load thetaInv_km1
        if(blockrow!=LEAD_BLOCK){
            loadblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                d_Pinv,
                s_thetaInv_km1,
                1,
                blockrow-1
            );
        }

        __syncthreads();//----------------------------------------------------------------

        // load thetaInv_kp1
        if(blockrow!=LAST_BLOCK){
            loadblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                d_Pinv,
                s_thetaInv_kp1,
                1,
                blockrow+1
            );
        }
        

        __syncthreads();//----------------------------------------------------------------

        if(blockrow!=LEAD_BLOCK){

            // compute left off diag    
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, false>(
                s_thetaInv_k,
                s_phi_k,
                s_scratch                                     
            );
            __syncthreads();//----------------------------------------------------------------
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, false>(
                s_scratch,
                s_thetaInv_km1,
                s_PhiInv_k_L 
            );
            __syncthreads();//----------------------------------------------------------------

            // store left diagonal in Phi
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_PhiInv_k_L, 
                d_Pinv,
                0,
                blockrow,
                -1
            );
            __syncthreads();//----------------------------------------------------------------
        }


        if(blockrow!=LAST_BLOCK){

            // should do transpose here
            // calculate Phi right diag
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, false>(
                s_thetaInv_k,
                s_phi_kp1_T,
                s_scratch                                     
            );
            __syncthreads();//----------------------------------------------------------------
            mat_mat_prod<c_float, STATE_SIZE, STATE_SIZE, STATE_SIZE, STATE_SIZE, false>(
                s_scratch,
                s_thetaInv_kp1,
                s_PhiInv_k_R
            );
            __syncthreads();//----------------------------------------------------------------


            // store Phi right diag
            storeblock_funnyformat<c_float, STATE_SIZE, KNOT_POINTS>(
                s_PhiInv_k_R, 
                d_Pinv,
                2,
                blockrow,
                -1
            );
        }
    }


    __global__
    void gato_form_ss(c_float *d_S, c_float *d_Pinv, c_float *d_gamma){
        
        const unsigned s_temp_size = 9 * STATES_SQ;
        // 8 * states^2
        // scratch space = states^2

        __shared__ c_float s_temp[ s_temp_size ];

        for(unsigned ind=GATO_BLOCK_ID; ind<KNOT_POINTS; ind+=GATO_NUM_BLOCKS){
            gato_form_ss_inner(
                s_temp,
                d_S,
                d_Pinv,
                d_gamma,
                ind
            );
        }
    }

#endif /* #if SS_PRECON */

    __global__
    void gato_form_schur_jacobi(c_float *d_G,
                                c_float *d_C,
                                c_float *d_g,
                                c_float *d_S,
                                c_float *d_Pinv, 
                                c_float *d_gamma){


        const unsigned s_temp_size =    10 * STATE_SIZE*STATE_SIZE+   
                                        5 * STATE_SIZE+ 
                                        STATE_SIZE * CONTROL_SIZE+
                                        6 * STATE_SIZE + 3;
                                    // TODO: determine actual shared mem size needed
        
        __shared__ c_float s_temp[ s_temp_size ];

        for(unsigned blockrow=GATO_BLOCK_ID; blockrow<KNOT_POINTS; blockrow+=GATO_NUM_BLOCKS){

            gato_form_schur_jacobi_inner(
                d_G,
                d_C,
                d_g,
                d_S,
                d_Pinv,
                d_gamma,
                s_temp,
                blockrow
            );
        
        }
    }

    


    /*******************************************************************************
    *                        private functions pcg solve                           *
    *******************************************************************************/

    template<typename T, unsigned PRECONDITIONER_BANDWITH = 3, unsigned N, bool USE_TRACE = false>
    __device__
    void parallelPCG_inner_fixed(c_float *d_S, c_float *d_pinv, c_float *d_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new, c_float *d_r_tilde, c_float *d_upsilon,	// global vectors and scalars
                            c_float *s_temp, T exitTol, unsigned maxIters, unsigned *iters 	    // shared mem for use in CG step and constants
                            cgrps::thread_block block, cgrps::grid_group grid){
                                
        //Initialise shared memory
        c_float *s_lambda = s_temp;
        c_float *s_r_tilde = s_lambda + STATE_SIZE;
        c_float *s_upsilon = s_r_tilde + STATE_SIZE;
        c_float *s_v_b = s_upsilon + STATE_SIZE;
        c_float *s_eta_new_b = s_v_b + STATE_SIZE;

        c_float *s_r = s_eta_new_b + STATE_SIZE;
        c_float *s_p = s_r + 3*STATE_SIZE;

        c_float *s_r_b = s_r + STATE_SIZE;
        c_float *s_p_b = s_p + STATE_SIZE;

        T alpha, beta;
        T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

        // Need to initialise *s_S, c_float *s_pinv, c_float *s_gamma and input all required as dynamic memory
        c_float *s_S = s_p + 3 * STATE_SIZE;
        c_float *s_pinv = s_S + 3*STATE_SIZE*STATE_SIZE;
        c_float *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;

        // Used when writing to device memory
        int bIndStateSize;

        if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            *iters = maxIters;
        }

        // Initililiasation before the main-pcg loop
        // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
        // Therefore, s_r_b = s_gamma_b

        for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){

            // directly write to device memory
            bIndStateSize = STATE_SIZE * block_number;
            // We find the s_r, load it into device memory, initialise lambda to 0
            for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                d_r[bIndStateSize + ind] = d_gamma[STATE_SIZE * block_number + ind]; 
                d_lambda[bIndStateSize + ind] = static_cast<T>(0);
            }

            block.sync();
        }

        if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
            *d_eta_new = static_cast<T>(0);

            //TODO:remove redundant
            *d_v = static_cast<T>(0);
        }
        // Need to sync before loading from other blocks
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        // if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
        //         print_raw_shared_vector<c_float,STATE_SIZE*N>(d_r);
        // }
        // grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        
        
        for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){
            // load s_r_b, pinv
            bIndStateSize = STATE_SIZE * block_number;
            for (unsigned ind= GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind]; 
            }
            loadBlockTriDiagonal_offDiagonal_fixed<c_float,STATE_SIZE, N>(s_r,&d_r[bIndStateSize],block_number,block,grid);
            block.sync();
            matVecMultBlockTriDiagonal_fixed<c_float,STATE_SIZE, N>(s_r_tilde,s_pinv,s_r,block_number,block,grid);
            block.sync();

            dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
            block.sync();

            // We copy p from r_tilde and write to device, since it will be required by other blocks
            //write back s_r_tilde, s_p
            for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                d_p[bIndStateSize + ind] = s_r_tilde[ind];
                d_r_tilde[bIndStateSize + ind] = s_r_tilde[ind];
            }
            if(GATO_LEAD_THREAD){
                // printf("Partial sums of Block %d and Block Number %d: %f\n", GATO_BLOCK_ID, block_number,s_eta_new_b[0] );
                atomicAdd(d_eta_new,s_eta_new_b[0]);
            }
            block.sync();
        
            
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        eta = *d_eta_new;
        block.sync();
        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
        // }
        for(unsigned iter = 0; iter < maxIters; iter++){
            if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
                *d_v = static_cast<T>(0);
            }
            grid.sync();
            // for over rows, 
            for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){

                bIndStateSize = STATE_SIZE * block_number;
                // s_S, s_p (already) load from device for that particular row
                for (unsigned ind = GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_S[ind] = d_S[bIndStateSize * STATE_SIZE * 3 + ind]; 
                }
                block.sync();


                loadBlockTriDiagonal_offDiagonal_fixed<c_float,STATE_SIZE,N>(s_p,&d_p[bIndStateSize],block_number,block,grid);
                block.sync();
                matVecMultBlockTriDiagonal_fixed<c_float,STATE_SIZE,N>(s_upsilon,s_S,s_p,block_number,block,grid);
                block.sync();
                dotProd<c_float,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
                block.sync();

                // only upsilon needs to be written to device memory
                for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    d_upsilon[bIndStateSize + ind] = s_upsilon[ind];
                }

                if(GATO_LEAD_THREAD){
                    atomicAdd(d_v,s_v_b[0]);
                }
                block.sync();
                
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){

                bIndStateSize = STATE_SIZE * block_number;
                
                // load s_p, s_lambda, s_upsilon, s_r
                for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_p_b[ind] = d_p[bIndStateSize + ind];
                    s_lambda[ind] = d_lambda[bIndStateSize + ind];
                    s_upsilon[ind] = d_upsilon[bIndStateSize + ind];
                    s_r_b[ind] = d_r[bIndStateSize + ind];
                }

                alpha = eta / *d_v;

                //Dont need this
                block.sync();

                // Move this loop into a function, write  back lambda and r
                for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_lambda[ind] += alpha * s_p_b[ind];
                    s_r_b[ind] -= alpha * s_upsilon[ind];
                    d_lambda[bIndStateSize + ind] = s_lambda[ind];
                    d_r[bIndStateSize + ind] = s_r_b[ind];
                }
                block.sync();
            }

            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            
            if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
                *d_eta_new = static_cast<T>(0);
            }
            block.sync();
            for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){

                bIndStateSize = STATE_SIZE * block_number;
                // load s_r (already), s_pinv
                for (unsigned ind = GATO_THREAD_ID; ind < 3*STATE_SIZE*STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_pinv[ind] = d_pinv[bIndStateSize * STATE_SIZE * 3 + ind]; 
                }

                loadBlockTriDiagonal_offDiagonal_fixed<c_float,STATE_SIZE,N>(s_r,&d_r[bIndStateSize],block_number,block,grid);
                block.sync();
                matVecMultBlockTriDiagonal_fixed<c_float,STATE_SIZE,N>(s_r_tilde,s_pinv,s_r,block_number,block,grid);
                block.sync();
                dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_tilde,s_r_b,block);
                block.sync();
                // write back r_tilde
                for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    d_r_tilde[bIndStateSize + ind] = s_r_tilde[ind];
                }

                if(GATO_LEAD_THREAD){
                    atomicAdd(d_eta_new,s_eta_new_b[0]);
                }
                block.sync();
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            
            // test for exit
            eta_new = *d_eta_new;
            
            if(abs(eta_new) < exitTol){
                // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
                //     printf("Breaking at iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
                // }
                if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
                    *iters = iter;
                }
                
                break;
            }
            
            // else compute d_p for next loop
            else{

                beta = eta_new / eta;
                for( unsigned block_number = GATO_BLOCK_ID; block_number < N; block_number += GATO_NUM_BLOCKS){

                    bIndStateSize = STATE_SIZE * block_number;
                    // load s_p, s_r_tilde, write back s_p
                    for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                        s_p_b[ind] = d_p[bIndStateSize + ind];
                        s_r_tilde[ind] = d_r_tilde[bIndStateSize + ind];
                        s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
                        d_p[bIndStateSize + ind] = s_p_b[ind];
                    }  
                
                }
                eta = eta_new;
                block.sync();
            }
            // then global sync for next loop
            // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
            // }
            // then global sync for next loop
            
            grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
            
        } 
        
    }

    template<typename T, unsigned PRECONDITIONER_BANDWITH = 3, unsigned N, bool USE_TRACE = false>
    __global__
    void parallelPCG_fixed(c_float *d_S, c_float *d_pinv, c_float *d_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new, c_float *d_r_tilde, c_float *d_upsilon,	// global vectors and scalars
                            T exitTol=1e-6, unsigned maxIters=100,			    // shared mem for use in CG step and constants 
                            unsigned *iters
                            ){

        __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 11 * STATE_SIZE];

        cgrps::thread_block block = cgrps::this_thread_block();	 
        cgrps::grid_group grid = cgrps::this_grid();
        
        grid.sync();
        parallelPCG_inner_fixed<float, STATE_SIZE, PRECONDITIONER_BANDWITH, N, false>(d_S, d_pinv, d_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, d_r_tilde, d_upsilon, s_temp, 1e-4, 100, iters, block, grid);
        grid.sync();
    }

    template <typename T, unsigned PRECONDITIONER_BANDWITH = 3, bool USE_TRACE = false>
    __device__
    void parallelPCG_inner(c_float *s_S, c_float *s_pinv, c_float *s_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new,	// global vectors and scalars
                            c_float *s_temp, T exitTol, unsigned maxIters, unsigned *iters,			    // shared mem for use in CG step and constants
                            cgrps::thread_block block, cgrps::grid_group grid){                      
        //Initialise shared memory
        c_float *s_lambda = s_temp;
        c_float *s_r_tilde = s_lambda + STATE_SIZE;
        c_float *s_upsilon = s_r_tilde + STATE_SIZE;
        c_float *s_v_b = s_upsilon + STATE_SIZE;
        c_float *s_eta_new_b = s_v_b + STATE_SIZE;

        c_float *s_r = s_eta_new_b + STATE_SIZE;
        c_float *s_p = s_r + 3*STATE_SIZE;

        c_float *s_r_b = s_r + STATE_SIZE;
        c_float *s_p_b = s_p + STATE_SIZE;

        T alpha, beta;
        T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

        // Used when writing to device memory
        int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;

        // Initililiasation before the main-pcg loop
        // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
        // Therefore, s_r_b = s_gamma_b

        // We find the s_r, load it into device memory, initialise lambda to 0
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_r_b[ind] = s_gamma[ind];
            d_r[bIndStateSize + ind] = s_r_b[ind]; 
            s_lambda[ind] = static_cast<T>(0);
        }
        // Make eta_new zero
        if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
            *d_eta_new = static_cast<T>(0);
            *d_v = static_cast<T>(0);
        }

        if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            *iters = maxIters;
        }
        // Need to sync before loading from other blocks
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        loadBlockTriDiagonal_offDiagonal<c_float,STATE_SIZE>(s_r,&d_r[bIndStateSize],block,grid);
        block.sync();
        matVecMultBlockTriDiagonal<c_float,STATE_SIZE>(s_r_tilde,s_pinv,s_r,block,grid);
        block.sync();

        // We copy p from r_tilde and write to device, since it will be required by other blocks
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_p_b[ind] = s_r_tilde[ind];
            d_p[bIndStateSize + ind] = s_p_b[ind]; 
        }

        dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
        block.sync();

        if(GATO_LEAD_THREAD){
            // printf("Partial sums of Block %d: %f\n", GATO_BLOCK_ID, s_eta_new_b[0] );
            atomicAdd(d_eta_new,s_eta_new_b[0]);
        }

        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        eta = *d_eta_new;
        block.sync();

        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
        // }
        
        for(unsigned iter = 0; iter < maxIters; iter++){
            loadBlockTriDiagonal_offDiagonal<c_float,STATE_SIZE>(s_p,&d_p[bIndStateSize],block,grid);
            block.sync();
            matVecMultBlockTriDiagonal<c_float,STATE_SIZE>(s_upsilon,s_S,s_p,block,grid);
            block.sync();
            dotProd<c_float,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
            block.sync();

            if(GATO_LEAD_THREAD){
                atomicAdd(d_v,s_v_b[0]);
                // Ideally move to just before calculation but then needs extra sync
                if(GATO_LEAD_BLOCK){
                    *d_eta_new = static_cast<T>(0);
                }
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            alpha = eta / *d_v;

            block.sync();
            if(false){
                printf("d_pSp[%f] -> alpha[%f]\n",*d_v,alpha);
            }

            block.sync();

            // Move this loop into a function
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_lambda[ind] += alpha * s_p_b[ind];
                s_r_b[ind] -= alpha * s_upsilon[ind];
                d_r[bIndStateSize + ind] = s_r_b[ind];
                }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER

            loadBlockTriDiagonal_offDiagonal<c_float,STATE_SIZE>(s_r,&d_r[bIndStateSize],block,grid);
            block.sync();
            matVecMultBlockTriDiagonal<c_float,STATE_SIZE>(s_r_tilde,s_pinv,s_r,block,grid);
            block.sync();
            dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_tilde,block);
            block.sync();
            if(GATO_LEAD_THREAD){
                atomicAdd(d_eta_new,s_eta_new_b[0]);
                // Ideally move to just before calculation but then needs extra sync
                if(GATO_LEAD_BLOCK){
                    *d_v = static_cast<T>(0);
                }
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            eta_new = *d_eta_new;

            block.sync();
            if(false){
                printf("eta_new[%f]\n",eta_new);
            }
            block.sync();

            if(abs(eta_new) < exitTol){

                if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
                    *iters = iter;
                }

                break;
            }
            
            // else compute d_p for next loop
            else{
                beta = eta_new / eta;
                for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
                    d_p[bIndStateSize + ind] = s_p_b[ind];
                }
                eta = eta_new;
            }

            // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
            // }
            // then global sync for next loop
            grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
            
        }
        // save final lambda to global
        block.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            d_lambda[bIndStateSize + ind] = s_lambda[ind];
        }
        
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    }

    template <typename T, unsigned PRECONDITIONER_BANDWITH = 3, bool USE_TRACE = false>
    __global__
    void parallelPCG(c_float *d_S, c_float *d_pinv, c_float *d_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new,	// global vectors and scalars
                            T exitTol = 1e-6, unsigned maxIters=100, unsigned *iters	// shared mem for use in CG step and constants
                            ){

        __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 11 * STATE_SIZE];
        c_float *s_S = s_temp;
        c_float *s_pinv = s_S +3*STATE_SIZE*STATE_SIZE;
        c_float *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;
        c_float *shared_mem = s_gamma + STATE_SIZE;

        cgrps::thread_block block = cgrps::this_thread_block();	 
        cgrps::grid_group grid = cgrps::this_grid();

        int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;
        for (unsigned ind = GATO_THREAD_ID; ind < 3 * STATE_SIZE * STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_S[ind] = d_S[bIndStateSize*STATE_SIZE*3 + ind];
            s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind];
        }
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_gamma[ind] = d_gamma[bIndStateSize + ind];
        }
        grid.sync();
        //Fix maxiter and exitTol issue
        parallelPCG_inner<float, STATE_SIZE, PRECONDITIONER_BANDWITH, false>(s_S, s_pinv, s_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, shared_mem, 1e-4, 100, iters, block, grid);
        grid.sync();
    }


    template <typename T, unsigned PRECONDITIONER_BANDWITH = 3, bool USE_TRACE = false>
    __device__
    void parallelCG_inner(c_float *s_S, c_float *s_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new,	// global vectors and scalars
                            c_float *s_temp, T exitTol, unsigned maxIters,			    // shared mem for use in CG step and constants
                            cgrps::thread_block block, cgrps::grid_group grid){                      
        //Initialise shared memory
        c_float *s_lambda = s_temp;
        c_float *s_upsilon = s_lambda+ STATE_SIZE;
        c_float *s_v_b = s_upsilon + STATE_SIZE;
        c_float *s_eta_new_b = s_v_b + STATE_SIZE;

        c_float *s_r = s_eta_new_b + STATE_SIZE;
        c_float *s_p = s_r + 3*STATE_SIZE;

        c_float *s_r_b = s_r + STATE_SIZE;
        c_float *s_p_b = s_p + STATE_SIZE;

        T alpha, beta;
        T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

        // Used when writing to device memory
        int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;

        // Initililiasation before the main-pcg loop
        // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
        // Therefore, s_r_b = s_gamma_b

        // We find the s_r, load it into device memory, initialise lambda to 0
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_r_b[ind] = s_gamma[ind];
            d_r[bIndStateSize + ind] = s_r_b[ind]; 
            s_lambda[ind] = static_cast<T>(0);
        }
        // Make eta_new zero
        if(GATO_LEAD_THREAD && GATO_LEAD_BLOCK){
            *d_eta_new = static_cast<T>(0);
            *d_v = static_cast<T>(0);
        }

        // Need to sync before loading from other blocks
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER


        // We copy p from r_tilde and write to device, since it will be required by other blocks
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_p_b[ind] = s_r_b[ind];
            d_p[bIndStateSize + ind] = s_p_b[ind]; 
        }

        dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_b,block);
        block.sync();

        if(GATO_LEAD_THREAD){
            // printf("Partial sums of Block %d: %f\n", GATO_BLOCK_ID, s_eta_new_b[0] );
            atomicAdd(d_eta_new,s_eta_new_b[0]);
        }

        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        eta = *d_eta_new;
        block.sync();

        // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
        //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
        // }
        
        for(unsigned iter = 0; iter < maxIters; iter++){
            loadBlockTriDiagonal_offDiagonal<c_float,STATE_SIZE>(s_p,&d_p[bIndStateSize],block,grid);
            block.sync();
            matVecMultBlockTriDiagonal<c_float,STATE_SIZE>(s_upsilon,s_S,s_p,block,grid);
            block.sync();
            dotProd<c_float,STATE_SIZE>(s_v_b,s_p_b,s_upsilon,block);
            block.sync();

            if(GATO_LEAD_THREAD){
                atomicAdd(d_v,s_v_b[0]);
                // Ideally move to just before calculation but then needs extra sync
                if(GATO_LEAD_BLOCK){
                    *d_eta_new = static_cast<T>(0);
                }
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            alpha = eta / *d_v;

            block.sync();
            if(false){
                printf("d_pSp[%f] -> alpha[%f]\n",*d_v,alpha);
            }

            block.sync();

            // Move this loop into a function
            for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                s_lambda[ind] += alpha * s_p_b[ind];
                s_r_b[ind] -= alpha * s_upsilon[ind];
                d_r[bIndStateSize + ind] = s_r_b[ind];
                }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER


            dotProd<c_float,STATE_SIZE>(s_eta_new_b,s_r_b,s_r_b,block);
            block.sync();
            if(GATO_LEAD_THREAD){
                atomicAdd(d_eta_new,s_eta_new_b[0]);
                // Ideally move to just before calculation but then needs extra sync
                if(GATO_LEAD_BLOCK){
                    *d_v = static_cast<T>(0);
                }
            }
            grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
            eta_new = *d_eta_new;

            block.sync();
            if(false){
                printf("eta_new[%f]\n",eta_new);
            }
            block.sync();

            if(abs(eta_new) < exitTol){
                break;
            }
            
            // else compute d_p for next loop
            else{
                beta = eta_new / eta;
                for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
                    s_p_b[ind] = s_r_b[ind] + beta*s_p_b[ind];
                    d_p[bIndStateSize + ind] = s_p_b[ind];
                }
                eta = eta_new;
            }

            // if(GATO_LEAD_BLOCK && GATO_LEAD_THREAD){
            //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
            // }
            // then global sync for next loop
            grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
            
        }
        // save final lambda to global
        block.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        for(unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            d_lambda[bIndStateSize + ind] = s_lambda[ind];
        }
        
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    }

    template <typename T, unsigned PRECONDITIONER_BANDWITH = 3, bool USE_TRACE = false>
    __global__
    void parallelCG(c_float *d_S, c_float *d_pinv, c_float *d_gamma,  				// block-local constant temporary variable inputs
                            c_float *d_lambda, c_float *d_r, c_float *d_p, c_float *d_v, c_float *d_eta_new,	// global vectors and scalars
                            T exitTol = 1e-6, unsigned maxIters=100			    // shared mem for use in CG step and constants
                            ){

        __shared__ T s_temp[3*STATE_SIZE*STATE_SIZE + 3*STATE_SIZE*STATE_SIZE + STATE_SIZE + 10 * STATE_SIZE];
        c_float *s_S = s_temp;
        c_float *s_pinv = s_S +3*STATE_SIZE*STATE_SIZE;
        c_float *s_gamma = s_pinv + 3*STATE_SIZE*STATE_SIZE;
        c_float *shared_mem = s_gamma + STATE_SIZE;

        cgrps::thread_block block = cgrps::this_thread_block();	 
        cgrps::grid_group grid = cgrps::this_grid();

        int bIndStateSize = STATE_SIZE * GATO_BLOCK_ID;
        for (unsigned ind = GATO_THREAD_ID; ind < 3 * STATE_SIZE * STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_S[ind] = d_S[bIndStateSize*STATE_SIZE*3 + ind];
            s_pinv[ind] = d_pinv[bIndStateSize*STATE_SIZE*3 + ind];
        }
        for (unsigned ind = GATO_THREAD_ID; ind < STATE_SIZE; ind += GATO_THREADS_PER_BLOCK){
            s_gamma[ind] = d_gamma[bIndStateSize + ind];
        }
        grid.sync();
        //Fix maxiter and exitTol issue
        parallelCG_inner<float, STATE_SIZE, PRECONDITIONER_BANDWITH, false>(s_S, s_gamma, d_lambda, d_r, d_p, d_v, d_eta_new, shared_mem, 1e-4, 100, block, grid);
        grid.sync();
    }

    /*******************************************************************************
    *                                   API                                        *
    *******************************************************************************/
        

    template <typename T, unsigned N>
    int check_sms(void* kernel, dim3 block){
        int dev = 0;
        cudaDeviceProp deviceProp; 
        cudaGetDeviceProperties(&deviceProp, dev);
        int supportsCoopLaunch = 0; 
        cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
        if(!supportsCoopLaunch){
            // printf("[Error] Device does not support Cooperative Threads -- this code will fail!\n");
            return 0;
        }
        int numProcs = static_cast<T>(deviceProp.multiProcessorCount); 
        int numBlocksPerSm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x*block.y*block.z, 0);
        if(N > numProcs*numBlocksPerSm){
            //printf("Too many KNOT_POINTS. Device supports [%d] blocks, [%d] SMs. Use the new algo\n",numProcs*numBlocksPerSm, numProcs);
            return numProcs*numBlocksPerSm;
        }
        else{
            //printf("Sufficient blocks for given KNOT_POINTS. Device supports [%d] blocks, [%d] SMs. Use the old algo\n",numProcs*numBlocksPerSm, numProcs);
            return N;
        }
    }



    template <typename T, unsigned N, unsigned PRECONDITIONER_BANDWITH=3>
    c_int solve_pcg(c_float *d_S, c_float *d_Pinv, c_float *d_gamma, f_float *d_lambda, c_float eps, c_int max_iter){

        c_float *d_r, *d_p, *d_v, *d_eta_new, *d_r_tilde, *d_upsilon;
        unsigned *iters;
        
        float exitTol = 1e-4;
        unsigned maxIters = 100;
        // TODO: enable these
        // unsigned maxIters = max_iter;
        // float exitTol = eps;
        unsigned sharedMemSize;

        
        
        cudaMallocManaged(&d_r, STATE_SIZE*N*sizeof(T));
        cudaMallocManaged(&d_p, STATE_SIZE*N*sizeof(T));
        cudaMallocManaged(&d_v, sizeof(T));
        cudaMallocManaged(&d_eta_new, sizeof(T));
        cudaMallocManaged(&d_r_tilde, STATE_SIZE*N*sizeof(T));
        cudaMallocManaged(&d_upsilon, STATE_SIZE*N*sizeof(T));
        cudaMallocManaged(&iters, sizeof(unsigned));

        
        sharedMemSize = (2 * 3 * STATES_SQ + STATE_SIZE + 11 * STATE_SIZE)*sizeof(T);
        
        dim3 grid(N,1,1);
        dim3 block(STATE_SIZE,1,1);
        cudaDeviceSynchronize();
        
    
        void *my_kernel = (void *)parallelPCG<c_float, STATE_SIZE, PRECONDITIONER_BANDWITH, false>;
        int num_blocks = check_sms<c_float,N>(my_kernel, block);
        cudaDeviceSynchronize();
        //Each block does exactly one row
        if(false){
            void *kernelArgsCG[] = {
                (void *)&d_S,
                (void *)&d_Pinv,
                (void *)&d_gamma, 
                (void *)&d_lambda,
                (void *)&d_r,
                (void *)&d_p,
                (void *)&d_v,
                (void *)&d_eta_new,
                (void *)&maxIters,
                (void *)&exitTol
            };
            // printf("Using the old algo \n");
            void *cg = (void *)parallelCG<c_float, STATE_SIZE, PRECONDITIONER_BANDWITH, false>;
            sharedMemSize = (1*3 * STATES_SQ + STATE_SIZE + 10 * STATE_SIZE)*sizeof(T);
            cudaLaunchCooperativeKernel(cg, grid, block, kernelArgsCG, sharedMemSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors( cudaPeekAtLastError() );
        }
        else if(num_blocks == N){
            void *kernelArgs[] = {
                (void *)&d_S,
                (void *)&d_Pinv,
                (void *)&d_gamma, 
                (void *)&d_lambda,
                (void *)&d_r,
                (void *)&d_p,
                (void *)&d_v,
                (void *)&d_eta_new,
                (void *)&maxIters,
                (void *)&exitTol,
                (void *)&iters
            };
            // printf("Using the old algo \n");
            cudaLaunchCooperativeKernel(my_kernel, grid, block, kernelArgs, sharedMemSize);
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors( cudaPeekAtLastError() );
        }

        //Each blocks needs to do more rows
        else if(num_blocks < N){
            // printf("Using the new algo \n");
            void *kernelArgsFixed[] = {
                (void *)&d_S,
                (void *)&d_Pinv,
                (void *)&d_gamma, 
                (void *)&d_lambda,
                (void *)&d_r,
                (void *)&d_p,
                (void *)&d_v,
                (void *)&d_eta_new,
                (void *)&d_r_tilde,
                (void *)&d_upsilon,
                (void *)&maxIters,
                (void *)&exitTol,
                (void *)&iters
            };
            dim3 grid_fixed(num_blocks,1,1);
            void *my_kernel_fixed = (void *)parallelPCG_fixed<c_float, STATE_SIZE, PRECONDITIONER_BANDWITH, N, false>;
            checkCudaErrors(cudaLaunchCooperativeKernel(my_kernel_fixed, grid_fixed, block, kernelArgsFixed, sharedMemSize));
            checkCudaErrors( cudaPeekAtLastError() );
            checkCudaErrors( cudaDeviceSynchronize() );
        }
        else{
            return 1;
        }

        // printf("Final Answer: \n");

        // for (int i = 0; i < STATE_SIZE*N; i++){
        //     std::cout<<d_lambda[i]<<"\n";
        // }

        
        cudaFree(d_r);
        cudaFree(d_p);
        cudaFree(d_v);
        cudaFree(d_eta_new);
        cudaFree(d_r_tilde);
        cudaFree(d_upsilon);

        unsigned _iters = *iters;
        cudaFree(iters);

        return _iters;
    }

}   /* namespace gato */

