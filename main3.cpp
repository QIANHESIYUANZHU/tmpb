#include "Defines.h"
#include <omp.h>
#include <immintrin.h>


inline void correntesch_store(ComplexType result1, ComplexType result2, ComplexType result3) {
	double re_diff, im_diff;

	re_diff = fabs(result1.real() - -264241151.454552);
	im_diff = fabs(result1.imag() - 1321205770.975190);
	re_diff += fabs(result2.real() - -137405397.758745);
	im_diff += fabs(result2.imag() - 961837795.884157);
	re_diff += fabs(result3.real() - -83783779.241634);
	im_diff += fabs(result3.imag() - 754054017.424472);
	printf("%f,%f\n",re_diff,im_diff);
	printf("%f, %f\n", fabs(-264241151.454552 + -137405397.758745 + -83783779.241634) * 1e-6, fabs(1321205770.975190 + 961837795.884157 + 754054017.424472) * 1e-6);

	if (re_diff < fabs(result1.real() + result2.real() + result3.real()) * 1e-6 && im_diff < fabs(result1.imag() + result2.imag() + result3.imag()) * 1e-6)
	{
		printf("\n!!!! SUCCEsch_store - !!!! Correctnesch_store test pasch_storeed :-D :-D\n\n");
	}
	else
	{
		printf("\n!!!! FAILURE - Correctnesch_store test failed :-( :-(  \n");
	}
}

int main(int argc, char **argv) {

	int number_bands = 0, nvband = 0, ncouls = 0, nodes_per_group = 0;
	int npes = 1;
	if (argc == 1) {
		number_bands = 512;
		nvband = 2;
		ncouls = 32768;
		nodes_per_group = 20;
	} else if (argc == 5) {
		number_bands = atoi(argv[1]);
		nvband = atoi(argv[2]);
		ncouls = atoi(argv[3]);
		nodes_per_group = atoi(argv[4]);
	} else {
		std::cout << "The correct form of input is : " << endl;
		std::cout << " ./main.exe <number_bands> <number_valence_bands> "
			"<number_plane_waves> <nodes_per_mpi_group> "
			<< endl;
		exit(0);
	}
	int ngpown = ncouls / (nodes_per_group * npes);

	// Constants that will be used later
	const DataType e_lk = 10;
	const DataType dw = 1;
	const DataType to1 = 1e-6;
	const DataType limittwo = pow(0.5, 2);
	const DataType e_n1kq = 6.0;

	// Using time point and system_clock
	time_point<system_clock> start, end, k_start, k_end;
	start = system_clock::now();
	double elapsedKernelTimer;

	// Printing out the params pasch_storeed.
	std::cout << "Sizeof(ComplexType = "
		<< sizeof(ComplexType) << " bytes" << std::endl;
	std::cout << "number_bands = " << number_bands << "\t nvband = " << nvband
		<< "\t ncouls = " << ncouls
		<< "\t nodes_per_group  = " << nodes_per_group
		<< "\t ngpown = " << ngpown << "\t nend = " << nend
		<< "\t nstart = " << nstart << endl;

	size_t memFootPrint = 0.00;

	// ALLOCATE statements .
	ARRAY1D achtemp(nend - nstart);
	memFootPrint += (nend - nstart) * sizeof(ComplexType);

	ARRAY2D aqsmtemp(number_bands, ncouls);
	ARRAY2D aqsntemp(number_bands, ncouls);
	memFootPrint += 2 * (number_bands * ncouls) * sizeof(ComplexType);

	ARRAY2D I_eps_array(ngpown, ncouls);
	ARRAY2D wtilde_array(ngpown, ncouls);
	memFootPrint += 2 * (ngpown * ncouls) * sizeof(ComplexType);

	ARRAY1D_DataType vcoul(ncouls);
	memFootPrint += ncouls * sizeof(DataType);

	ARRAY1D_int inv_igp_index(ngpown);
	ARRAY1D_int indinv(ncouls + 1);
	memFootPrint += ngpown * sizeof(int);
	memFootPrint += (ncouls + 1) * sizeof(int);

	ARRAY1D_DataType wx_array(nend - nstart);
	memFootPrint += 3 * (nend - nstart) * sizeof(DataType);

	// Print Memory Foot print
	cout << "Memory Foot Print = " << memFootPrint / pow(1024, 3) << " GBs"
		<< endl;

	ComplexType expr(.5, .5);
	for (int i = 0; i < number_bands; i++)
		for (int j = 0; j < ncouls; j++) {
			aqsmtemp(i, j) = expr;
			aqsntemp(i, j) = expr;
		}

	for (int i = 0; i < ngpown; i++)
		for (int j = 0; j < ncouls; j++) {
			I_eps_array(i, j) = expr;
			wtilde_array(i, j) = expr;
		}

	for (int i = 0; i < ncouls; i++)
		vcoul(i) = 1.0;

	for (int ig = 0; ig < ngpown; ++ig)
		inv_igp_index(ig) = (ig + 1) * ncouls / ngpown;

	for (int ig = 0; ig < ncouls; ++ig)
		indinv(ig) = ig;
	indinv(ncouls) = ncouls - 1;

	for (int iw = nstart; iw < nend; ++iw) {
		wx_array(iw) = e_lk - e_n1kq + dw * ((iw + 1) - 2);
		if (wx_array(iw) < to1)
			wx_array(iw) = to1;
	}

	k_start = system_clock::now();
	noflagOCC_solver(number_bands, ngpown, ncouls, inv_igp_index, indinv,
			wx_array, wtilde_array, aqsmtemp, aqsntemp, I_eps_array,
			vcoul, achtemp);

	k_end = system_clock::now();
	duration<double> elapsed = k_end - k_start;
	elapsedKernelTimer = elapsed.count();

	// Check for correctnesch_store
	correntesch_store(achtemp(0),achtemp(1),achtemp(2));	

	printf("\n Final achtemp\n");
	ComplexType_print(achtemp(0));
	ComplexType_print(achtemp(1));
	ComplexType_print(achtemp(2));

	end = system_clock::now();
	elapsed = end - start;

	cout << "********** Kernel Time Taken **********= " << elapsedKernelTimer
		<< " secs" << endl;
	cout << "********** Total Time Taken **********= " << elapsed.count()
		<< " secs" << endl;

	return 0;
}

void noflagOCC_solver(size_t number_bands, size_t ngpown, size_t ncouls,
                       ARRAY1D_int &inv_igp_index, ARRAY1D_int &indinv,
                       ARRAY1D_DataType &wx_array, ARRAY2D &wtilde_array,
                       ARRAY2D &aqsmtemp, ARRAY2D &aqsntemp,
                       ARRAY2D &I_eps_array, ARRAY1D_DataType &vcoul,
                       ARRAY1D &achtemp)
{
    time_point<system_clock> start, end;
    start = system_clock::now();

    DataType ach_re0 = 0.00, ach_re1 = 0.00, ach_re2 = 0.00, ach_im0 = 0.00,
             ach_im1 = 0.00, ach_im2 = 0.00;

    int n1, ig, igp, iw, my_igp;
    ARRAY2D sch_store1(ngpown,number_bands);
    ARRAY1D sch_store(ngpown);

//最后面有优化的各个版本的关键代码 
  
//循环分离技术，将多层循环拆开，减少循环嵌套层数，将多层循环分解成少层循环的串行结构，总的复杂度降低 V5 1.05s  代码如下：         
                     
for (n1 = 0; n1 < number_bands; ++n1)
    {
		
		for (my_igp = 0; my_igp < ngpown; ++my_igp)
             {
	            int iigp=indinv(inv_igp_index(my_igp));
	            //  int indigp = inv_igp_index(my_igp);
                //  int igp = indinv(indigp);
                sch_store1(my_igp,n1) = ComplexType_conj(aqsmtemp(n1, iigp)) * aqsntemp(n1, iigp) * 0.5 * vcoul(iigp) * wtilde_array(my_igp, iigp);
                sch_store(my_igp) += sch_store1(my_igp,n1);
               }           
     }
/*
for (my_igp = 0; my_igp < ngpown; ++my_igp)
{
	for (n1 = 0; n1 < number_bands; ++n1)
    {
		
		//sch_store(my_igp) += sch_store(my_igp,n1);
		}
	
	}*/

#pragma omp parallel for private(n1, ig, igp, iw, my_igp) \
                             reduction(+:ach_re0, ach_re1, ach_re2, ach_im0, ach_im1, ach_im2) \
                             schedule(dynamic) num_threads(64)
for (my_igp = 0; my_igp < ngpown; ++my_igp)
{
        
        for (ig = 0; ig < ncouls; ++ig)
        {
			//  DataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
			DataType __attribute__((aligned(8))) achtemp_re_loc[nend - nstart];
            DataType __attribute__((aligned(8))) achtemp_im_loc[nend - nstart];
                for (int i = nstart; i < nend; ++i)
                    {
                    achtemp_re_loc[i] = 0.00;
                    achtemp_im_loc[i] = 0.00;
                     }
             
            for (int iw = nstart; iw < nend; ++iw)
            {
                ComplexType wdiff = wx_array(iw) - wtilde_array(my_igp, ig);
                ComplexType delw = ComplexType_conj(wdiff) * (1 / (wdiff * ComplexType_conj(wdiff)).real());
                ComplexType sch_array = delw * I_eps_array(my_igp,ig)*sch_store(my_igp);
                achtemp_re_loc[iw] += (sch_array).real();
                achtemp_im_loc[iw] += (sch_array).imag();
            }
        
        #pragma omp atomic
        ach_re0 += achtemp_re_loc[0];
        #pragma omp atomic
        ach_re1 += achtemp_re_loc[1];
        #pragma omp atomic
        ach_re2 += achtemp_re_loc[2];
        #pragma omp atomic
        ach_im0 += achtemp_im_loc[0];
        #pragma omp atomic
        ach_im1 += achtemp_im_loc[1];
        #pragma omp atomic
        ach_im2 += achtemp_im_loc[2];
   } 
}

    achtemp(0) = ComplexType(ach_re0, ach_im0);
    achtemp(1) = ComplexType(ach_re1, ach_im1);
    achtemp(2) = ComplexType(ach_re2, ach_im2);
}




//循环顺序改变，减少计算量，按照循环嵌套原理，循环次数少的放外层，循环次数多的放外层  V1（但是不知道为什么将第一层循环和第二层循环交换顺序，反而性能提高。)  70s
/*  for (my_igp = 0; my_igp < ngpown; ++my_igp)
{
    for (n1 = 0; n1 < number_bands; ++n1)
    {
        int indigp = inv_igp_index(my_igp);
        int igp = indinv(indigp);
        DataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
        for (int iw = nstart; iw < nend; ++iw)
        {
            achtemp_re_loc[iw] = 0.00;
            achtemp_im_loc[iw] = 0.00;
        }
        ComplexType sch_store1 = ComplexType_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 * vcoul(igp) * wtilde_array(my_igp, igp);
        for (ig = 0; ig < ncouls; ++ig)
        {
            for (int iw = nstart; iw < nend; ++iw)
            {
                ComplexType wdiff = wx_array(iw) - wtilde_array(my_igp, ig);
                ComplexType delw = ComplexType_conj(wdiff) * (1 / (wdiff * ComplexType_conj(wdiff)).real());
                ComplexType sch_array = delw * I_eps_array(my_igp, ig) * sch_store1;
                achtemp_re_loc[iw] += (sch_array).real();
                achtemp_im_loc[iw] += (sch_array).imag();
            }
        }
        #pragma omp atomic
        ach_re0 += achtemp_re_loc[0];
        #pragma omp atomic
        ach_re1 += achtemp_re_loc[1];
        #pragma omp atomic
        ach_re2 += achtemp_re_loc[2];
        #pragma omp atomic
        ach_im0 += achtemp_im_loc[0];
        #pragma omp atomic
        ach_im1 += achtemp_im_loc[1];
        #pragma omp atomic
        ach_im2 += achtemp_im_loc[2];
    }
}

    achtemp(0) = ComplexType(ach_re0, ach_im0);
    achtemp(1) = ComplexType(ach_re1, ach_im1);
    achtemp(2) = ComplexType(ach_re2, ach_im2);
}*/

//循环展开技术   将最内层循环展开，减少重复运算v2 58s
/*  
 for (my_igp = 0; my_igp < ngpown; ++my_igp)
{
    for (n1 = 0; n1 < number_bands; ++n1)
    {
        int indigp = inv_igp_index(my_igp);
        int igp = indinv(indigp);
        DataType achtemp_re_loc[nend-nstart], achtemp_im_loc[nend-nstart];
        #pragma omp simd
        for(int i=0; i<(nend-nstart); i++)
        {
            achtemp_re_loc[i] = 0.0;
            achtemp_im_loc[i] = 0.0;
        }  
        ComplexType sch_store1 = ComplexType_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 * vcoul(igp) * wtilde_array(my_igp, igp);
        for(int i=nstart; i<nend; i+=4)
        {
            // Unroll the loop by a factor of 4
            for (ig = 0; ig < ncouls; ++ig)
            {
                ComplexType wdiff0 = wx_array(i) - wtilde_array(my_igp, ig);
                ComplexType wdiff1 = wx_array(i + 1) - wtilde_array(my_igp, ig);
                ComplexType wdiff2 = wx_array(i + 2) - wtilde_array(my_igp, ig);
                ComplexType wdiff3 = wx_array(i + 3) - wtilde_array(my_igp, ig);
                
                ComplexType delw0 = ComplexType_conj(wdiff0) * (1 / (wdiff0 * ComplexType_conj(wdiff0)).real());
                ComplexType delw1 = ComplexType_conj(wdiff1) * (1 / (wdiff1 * ComplexType_conj(wdiff1)).real());
                ComplexType delw2 = ComplexType_conj(wdiff2) * (1 / (wdiff2 * ComplexType_conj(wdiff2)).real());
                ComplexType delw3 = ComplexType_conj(wdiff3) * (1 / (wdiff3 * ComplexType_conj(wdiff3)).real());
                
                ComplexType sch_array0 = delw0 * I_eps_array(my_igp, ig) * sch_store1;
                ComplexType sch_array1 = delw1 * I_eps_array(my_igp, ig) * sch_store1;
                ComplexType sch_array2 = delw2 * I_eps_array(my_igp, ig) * sch_store1;
                ComplexType sch_array3 = delw3 * I_eps_array(my_igp, ig) * sch_store1;
                
                achtemp_re_loc[i-nstart] += (sch_array0).real();
                achtemp_re_loc[i-nstart+1] += (sch_array1).real();
                achtemp_re_loc[i-nstart+2] += (sch_array2).real();
                achtemp_re_loc[i-nstart+3] += (sch_array3).real();
                
                achtemp_im_loc[i-nstart] += (sch_array0).imag();
                achtemp_im_loc[i-nstart+1] += (sch_array1).imag();
                achtemp_im_loc[i-nstart+2] += (sch_array2).imag();
                achtemp_im_loc[i-nstart+3] += (sch_array3).imag();
            }
        }
        #pragma omp atomic
        ach_re0 += achtemp_re_loc[0];
        #pragma omp atomic
        ach_re1 += achtemp_re_loc[1];
        #pragma omp atomic
        ach_re2 += achtemp_re_loc[2];
        #pragma omp atomic
        ach_im0 += achtemp_im_loc[0];
        #pragma omp atomic
        ach_im1 += achtemp_im_loc[1];
        #pragma omp atomic
        ach_im2 += achtemp_im_loc[2];
    }
}



    achtemp(0) = ComplexType(ach_re0, ach_im0);
    achtemp(1) = ComplexType(ach_re1, ach_im1);
    achtemp(2) = ComplexType(ach_re2, ach_im2);
}
 */    
 
 
//循环展开基础上将最内层循环的类似计算合并成循环内V3   2.5s  
/*  
 for (my_igp = 0; my_igp < ngpown; ++my_igp)
	{
		for (n1 = 0; n1 < number_bands; ++n1)
		{
			int indigp = inv_igp_index(my_igp);
			int igp = indinv(indigp);
			DataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
			#pragma omp simd
			for (int i = 0; i < (nend - nstart); i++)
			{
				achtemp_re_loc[i] = 0.0;
				achtemp_im_loc[i] = 0.0;
			}
			ComplexType sch_store1 = ComplexType_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 * vcoul(igp) * wtilde_array(my_igp, igp);
			// Use vectorization to compute four elements at a time

			for (int i = nstart; i < nend; i += 4) 
			{
				for (ig = 0; ig < ncouls; ++ig)
				{
					DataType wdiff_re[4], wdiff_im[4], delw_re[4], delw_im[4], sch_array_re[4], sch_array_im[4];
					// Load four elements from wx_array and wtilde_array into wdiff
					#pragma omp simd
					for (int j = 0; j < 4; j++) {
						wdiff_re[j] = ((ComplexType)wx_array(i + j)).real() - ((ComplexType)wtilde_array(my_igp, j)).real();
						wdiff_im[j] = ((ComplexType)wx_array(i + j)).imag() - ((ComplexType)wtilde_array(my_igp, j)).imag();
						DataType norm_inv = 1 / (wdiff_re[j] * wdiff_re[j] + wdiff_im[j] * wdiff_im[j]);
						delw_re[j] = wdiff_re[j] * norm_inv;
						delw_im[j] = -wdiff_im[j] * norm_inv;
						DataType tmp_re = delw_re[j] * I_eps_array(my_igp, j).real() - delw_im[j] * I_eps_array(my_igp, j).imag();
						DataType tmp_im = delw_re[j] * I_eps_array(my_igp, j).imag() + delw_im[j] * I_eps_array(my_igp, j).real();
						delw_re[j] = tmp_re;
						delw_im[j] = tmp_im;
						sch_array_re[j] = delw_re[j] * sch_store1.real() - delw_im[j] * sch_store1.imag();
						sch_array_im[j] = delw_re[j] * sch_store1.imag() + delw_im[j] * sch_store1.real();
					}
					achtemp_re_loc[i - nstart] += sch_array_re[0];
					achtemp_re_loc[i - nstart + 1] += sch_array_re[1];
					achtemp_re_loc[i - nstart + 2] += sch_array_re[2];
					achtemp_re_loc[i - nstart + 3] += sch_array_re[3];

					achtemp_im_loc[i - nstart] += sch_array_im[0];
					achtemp_im_loc[i - nstart + 1] += sch_array_im[1];
					achtemp_im_loc[i - nstart + 2] += sch_array_im[2];
					achtemp_im_loc[i - nstart + 3] += sch_array_im[3];
				}
			}
#pragma omp atomic
			ach_re0 += achtemp_re_loc[0];
#pragma omp atomic
			ach_re1 += achtemp_re_loc[1];
#pragma omp atomic
			ach_re2 += achtemp_re_loc[2];
#pragma omp atomic
			ach_im0 += achtemp_im_loc[0];
#pragma omp atomic
			ach_im1 += achtemp_im_loc[1];
#pragma omp atomic
			ach_im2 += achtemp_im_loc[2];
		}
	}



	achtemp(0) = ComplexType(ach_re0, ach_im0);
	achtemp(1) = ComplexType(ach_re1, ach_im1);
	achtemp(2) = ComplexType(ach_re2, ach_im2);
}

 */        
 
 
//对变量以及复杂计算进行向量化，利用AVX256指令集 V4 未得出结果 结果比2.5s长，所以并未运行出结果就终止 

/* 
 for (my_igp = 0; my_igp < ngpown; my_igp += 1)
{
for (n1 = 0; n1 < number_bands; n1 += 1)
{
int indigp = inv_igp_index(my_igp);
int igp = indinv(indigp);
DataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
#pragma omp simd
for (int i = 0; i < (nend - nstart); i += 1)
{
achtemp_re_loc[i] = 0.0;
achtemp_im_loc[i] = 0.0;
}
ComplexType sch_store1 = ComplexType_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 * vcoul(igp) * wtilde_array(my_igp, igp);

    for (int i = nstart; i < nend; i += 4) 
    {
        for (ig = 0; ig < ncouls; ig += 1)
        {
            DataType wdiff_re, wdiff_im, delw_re, delw_im, sch_array_re, sch_array_im;
            wdiff_re = _mm256_set_pd(((ComplexType)wx_array(i + 3)).real(), ((ComplexType)wx_array(i + 2)).real(), ((ComplexType)wx_array(i + 1)).real(), ((ComplexType)wx_array(i)).real());
            wdiff_im = _mm256_set_pd(((ComplexType)wx_array(i + 3)).imag(), ((ComplexType)wx_array(i + 2)).imag(), ((ComplexType)wx_array(i + 1)).imag(), ((ComplexType)wx_array(i)).imag());
            DataType wtilde_re = _mm256_set1_pd(((ComplexType)wtilde_array(my_igp, ig)).real());
            DataType wtilde_im = _mm256_set1_pd(((ComplexType)wtilde_array(my_igp, ig)).imag());
            wdiff_re = _mm256_sub_pd(wdiff_re, wtilde_re);
            wdiff_im = _mm256_sub_pd(wdiff_im, wtilde_im);
            DataType norm_inv = _mm256_set1_pd(1);
            norm_inv = _mm256_div_pd(norm_inv, _mm256_add_pd(_mm256_mul_pd(wdiff_re, wdiff_re), _mm256_mul_pd(wdiff_im, wdiff_im)));
            delw_re = _mm256_mul_pd(wdiff_re, norm_inv);
            delw_im = _mm256_mul_pd(wdiff_im, norm_inv);
            DataType ie_d1_re = _mm256_set1_pd(I_eps_array(my_igp, ig).real());
            DataType ie_d1_im = _mm256_set1_pd(I_eps_array(my_igp, ig).imag());
            DataType tmp_re = _mm256_sub_pd(_mm256_mul_pd(delw_re, ie_d1_re), _mm256_mul_pd(delw_im, ie_d1_im));
            DataType tmp_im = _mm256_add_pd(_mm256_mul_pd(delw_re, ie_d1_im), _mm256_mul_pd(delw_im, ie_d1_re));
            delw_re = tmp_re;
            delw_im = tmp_im;
            sch_array_re = _mm256_sub_pd(_mm256_mul_pd(delw_re, _mm256_set1_pd(sch_store1.real())), _mm256_mul_pd(delw_im, _mm256_set1_pd(sch_store1.imag())));
            sch_array_im = _mm256_add_pd(_mm256_mul_pd(delw_re, _mm256_set1_pd(sch_store1.imag())), _mm256_mul_pd(delw_im, _mm256_set1_pd(sch_store1.real())));
            achtemp_re_loc[i - nstart] += sch_array_re[0];
            achtemp_re_loc[i - nstart + 1] += sch_array_re[1];
            achtemp_re_loc[i - nstart + 2] += sch_array_re[2];
            achtemp_re_loc[i - nstart + 3] += sch_array_re[3];
            achtemp_im_loc[i - nstart] += sch_array_im[0];
            achtemp_im_loc[i - nstart + 1] += sch_array_im[1];
            achtemp_im_loc[i - nstart + 2] += sch_array_im[2];
            achtemp_im_loc[i - nstart + 3] += sch_array_im[3];
        }
    }
    #pragma omp atomic
    ach_re0 += achtemp_re_loc[0];
    #pragma omp atomic
    ach_re1 += achtemp_re_loc[1];
    #pragma omp atomic
    ach_re2 += achtemp_re_loc[2];
    #pragma omp atomic
    ach_im0 += achtemp_im_loc[0];
    #pragma omp atomic
    ach_im1 += achtemp_im_loc[1];
    #pragma omp atomic
    ach_im2 += achtemp_im_loc[2];
}
}
achtemp(0) = ComplexType(ach_re0, ach_im0);
achtemp(1) = ComplexType(ach_re1, ach_im1);
achtemp(2) = ComplexType(ach_re2, ach_im2);
}

  */ 
