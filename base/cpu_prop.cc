#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include "papi.h"
#include <immintrin.h>
#include <zmmintrin.h>
#include "cpu_prop.h"
#include <algorithm>

#define C0  0
#define CZ1 1
#define CX1 2
#define CY1 3
#define CZ2 4
#define CX2 5
#define CY2 6
#define CZ3 7
#define CX3 8
#define CY3 9
#define CZ4 10
#define CX4 11
#define CY4 12

#define AVX_SIMD_LENGTH 16
#define ALIGNMENT 64
#define STRIDE 96
#define OFFSET 12
cpuProp::cpuProp(std::shared_ptr<SEP::genericIO> io){
	storeIO(io);

}

void cpuProp::rtmForward(int n1, int n2, int n3, int jt, float *img,
	float *rec, int npts, int nt, int nt_big, int rec_nx, int rec_ny){
    _jt=jt;
	std::vector<float> rec_p0(_n123,0.),rec_p1(_n123,0.);
	std::vector<float> src_p0(_n123,0.),src_p1(_n123,0.);
    float *r_p0=rec_p0.data(),*r_p1=rec_p1.data();
    float *s_p0=src_p0.data(),*s_p1=src_p1.data();


_dir=1;
	std::cout << "Calling the prop function (and other functions) " << nt << "times from rtmForward function" << std::endl;
for(int it=0; it < nt; it++){
    int id=it/jt;
    int ii=it-id*jt;
fprintf(stderr,"forward   %d of %d \n",it,nt);


    prop_naive(s_p0,s_p1,_vel2);
    prop_naive(r_p0,r_p1,_vel2);
    //stats(s_p0,"after prop");
    damp(s_p0,s_p1);
    damp(r_p0,r_p1);
    injectSource(id,ii,s_p0);

    imageAdd(img,r_p0,s_p0);
     
    dataExtract(id,ii,r_p0);


    float *pt=s_p0;
		s_p0=s_p1;
		s_p1=pt;
		pt=r_p0;
		r_p0=r_p1;
		r_p1=pt;
    }


}

void cpuProp::rtmAdjoint(int n1, int n2, int n3, int jtd, float *src_p0, float *src_p1,
	float *img, int npts_s, int nt){
        //rtm_adjoint(ad1.n,ad2.n,ad3.n,jtd,src_p0->vals,src_p1->vals,img->vals,npts_s,nt/*,src,recx*/);
	std::vector<float> rec_p0(_n123,0.),rec_p1(_n123,0.);
	float *r_p0=rec_p0.data(), *r_p1=rec_p1.data();
	//float *r_p0 = (float *)_mm_malloc((_n123)*sizeof(float), ALIGNMENT);
	//r_p0+=OFFSET;
	//float *r_p1 = (float *)_mm_malloc((_n123)*sizeof(float), ALIGNMENT);
	//r_p1+=OFFSET;
	
	_dir=-1;
	float sm1=0,sm2=0;
	for(int i=0; i < n1*n2*n3; i++){
		sm1=sm1+fabsf(src_p0[i]);
		sm2=sm2+fabsf(src_p1[i]);
	}

	int ic=0;

	for(int it=nt-1; it >=0; it--) {
		fprintf(stderr,"running adjoint %d  _nz=%d \n",it,_nz);
		int id_s=(it+1)/_jtsS;
		int i_s=it+1-id_s*_jtsS;
		int id=it/_jtdD;
		int ii=it-id*_jtdD;


		if(it>0) prop_naive(r_p0,r_p1,_vel2);
		if(it< nt-1) {
			prop_naive(src_p0,src_p1,_vel2);
			injectSource(id_s,i_s,src_p1);                                                                                                                                                                                                                                                                                                                                                                                                                               //I think this should be p0;
		}
		damp(r_p0,r_p1);
		injectReceivers(id,ii,r_p0);
		imageCondition(r_p0,src_p0,img);

		float *pt=src_p0;
		src_p0=src_p1;
		src_p1=pt;
		pt=r_p0;
		r_p0=r_p1;
		r_p1=pt;
		ic++;
	}
}

void cpuProp::imageCondition(const float *rec, const float *src, float *img){
 tbb::parallel_for(tbb::blocked_range<int>(0,_n123),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
		img[i]+=src[i]*rec[i];
   }

   });
}
void cpuProp::sourceProp(int nx, int ny, int nz, bool damp, bool getLast,
	float *p0, float *p1, int jts, int npts, int nt){

    _jt=jts;

	std::cout <<"The sourceProp functions calls the prop_vec function and injectSource function " << nt << " times" << std::endl;
	int n12=_nx*_ny;
	printf("nx = %d, ny = %d, nz = %d", _nx, _ny, _nz);
	_dir=1;
	for(int it=0; it<=nt; it++) {
		int id=it/_jtsS;
		int ii=it-id*_jtsS;
		
		printf("The time step is: %d\n",it);
		// prop_blocked(p0, blk_p1, _vel1, it)
		prop_naive(p0,p1,_vel1);

		injectSource(id,ii,p0);

		// copy_blocked(p0, blk_p1);
		float *pt=p1; p1=p0; p0=pt;	
	}
	if(nt%2==1){	
	std::cout << "Swapping the pointers" << std::endl;
	   float *x=new float[_n123];
	   memcpy(x,p0,sizeof(float)*_n123);
	   memcpy(p1,p0,sizeof(float)*_n123);
	   memcpy(p1,x,sizeof(float)*_n123);
	   //std::swap(p0, p1); 
	}

}
void cpuProp::stats(float *buf, std::string title){
float en= tbb::parallel_reduce(
        tbb::blocked_range<float*>( &buf[0], &buf[0]+_n123 ),
        0.f,
        [](const tbb::blocked_range<float*>& r, double init)->double {
            for( float* a=r.begin(); a!=r.end(); ++a )
                init += (*a)*(*a);
            return init;
        },
        []( double x, double y )->double {
            return x+y;
        }
    );

std::cerr<<title<<":STATS:"<<en<<std::endl;
}

void cpuProp::damp(float *p0,float *p1){

     tbb::parallel_for(tbb::blocked_range<int>(4,_nz-4),[&](
  const tbb::blocked_range<int>&r){
  for(int  i3=r.begin(); i3!=r.end(); ++i3){
		int edge1=std::min(i3-4,_nz-4-i3);
		for(int i2=4; i2 < _ny-4; i2++) {
			int edge2=std::min(edge1,std::min(i2-4,_ny-4-i2));
			int ii=i2*_nx+4+_n12*i3;
			for(int i1=4; i1 < _nx-4; i1++,ii++) {
				int edge=std::min(edge2,std::min(i1-4,_nx-4-i1));
	
				if(edge>=0 && edge < _bound.size()) {
					//		if(i2==600 && i1==600 && i3 < 70)
				 // fprintf(stderr,"check i3=%d edge=%d value=%f \n",
				 //x i3,edge,_bound[edge]); 
					p0[ii]*=_bound[edge];
					p1[ii]*=_bound[edge];
				}
			}
		}
	}
});
}
void cpuProp::injectSource(int id, int ii, float *p){
   
   if(id+7 >= _ntSrc)  return;
	for(int i=0; i < _nptsS; i++) {
		//std::cout << "The number of iteration of the inject Source function is "<< _nptsS << std::endl;
		p[_locsS[i]]+=_dir/(float)_jt*(
			_tableS[ii][0]*_sourceV[_ntSrc*i+id]+
			_tableS[ii][1]*_sourceV[_ntSrc*i+id+1]+
			_tableS[ii][2]*_sourceV[_ntSrc*i+id+2]+
			_tableS[ii][3]*_sourceV[_ntSrc*i+id+3]+
			_tableS[ii][4]*_sourceV[_ntSrc*i+id+4]+
			_tableS[ii][5]*_sourceV[_ntSrc*i+id+5]+
			_tableS[ii][6]*_sourceV[_ntSrc*i+id+6]+
			_tableS[ii][7]*_sourceV[_ntSrc*i+id+7]);
			

			
	}
	
}
void cpuProp::injectReceivers(int id, int ii, float *p){


   if(id+7 >= _ntRec)  return;
float sc=(float)_dir/(float) _jt;
     tbb::parallel_for(tbb::blocked_range<int>(0,_nRecs),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
		p[_locsR[i]]+=sc*(
			_tableD[ii][0]*_rec[_ntRec*i+id]+
			_tableD[ii][1]*_rec[_ntRec*i+id+1]+
			_tableD[ii][2]*_rec[_ntRec*i+id+2]+
			_tableD[ii][3]*_rec[_ntRec*i+id+3]+
			_tableD[ii][4]*_rec[_ntRec*i+id+4]+
			_tableD[ii][5]*_rec[_ntRec*i+id+5]+
			_tableD[ii][6]*_rec[_ntRec*i+id+6]+
			_tableD[ii][7]*_rec[_ntRec*i+id+7]);


	}
	});
}
void cpuProp::dataExtract(int id, int ii, const float *p){


   if(id+7 >= _ntRec)  return;

     tbb::parallel_for(tbb::blocked_range<int>(0,_nRecs),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
    _rec[_ntRec*i+id]+=p[_locsR[i]]*_tableD[ii][0];
    _rec[_ntRec*i+id+1]+=p[_locsR[i]]*_tableD[ii][1];
    _rec[_ntRec*i+id+2]+=p[_locsR[i]]*_tableD[ii][2];
    _rec[_ntRec*i+id+3]+=p[_locsR[i]]*_tableD[ii][3];
    _rec[_ntRec*i+id+4]+=p[_locsR[i]]*_tableD[ii][4];
    _rec[_ntRec*i+id+5]+=p[_locsR[i]]*_tableD[ii][5];
    _rec[_ntRec*i+id+6]+=p[_locsR[i]]*_tableD[ii][6];
    _rec[_ntRec*i+id+7]+=p[_locsR[i]]*_tableD[ii][7];
	}
	});
}
void cpuProp::imageAdd(const float *img,  float *recField, const float *srcField){
     tbb::parallel_for(tbb::blocked_range<int>(0,_n123),[&](
  const tbb::blocked_range<int>&r){
  for(int  i=r.begin(); i!=r.end(); ++i){
    recField[i]+=.000001*srcField[i]*img[i];
  }});
}

void cpuProp::copy_blocked(const float* p0, float* b1){  
	int stride = STRIDE;
	int offset = 4;
	
	int blk=0;
        int b_nx = (stride + 2*offset);
	int b_n12 = (stride + 2 *offset) * (stride + 2*offset);	
	int blk_sz = (stride + 2*offset) * (stride + 2*offset)  * (stride + 2*offset);
	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
	    for(int ii2=offset; ii2 < _ny - offset ; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			tbb::parallel_for(tbb::blocked_range<int>(ii3 - offset,std::min(ii3+stride + offset,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			int blk_ct = blk*blk_sz;
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
//			for (int i3 = ii3 - offset; i3 < std::min(ii3 + offset + stride,_nz); i3++){
			    int z_ct = (i3 - ii3 + offset);
		            int y_ct = 0;
			    for(int i2=ii2 - offset; i2 < std::min(ii2+stride+offset, _ny); i2++, y_ct++) {
			    	int bidx = blk_ct + y_ct*b_nx + z_ct * b_n12;
				int pidx = (i3) * _n12 + (i2) * _nx  + ii1 - offset; 
			    	for(int i1=ii1 - offset; i1 < std::min(ii1+stride+offset, _nx); i1++) {
					b1[bidx] = p0[pidx]; 
			//      	printf("p0[%d] = %f\n",pidx, p0[pidx]);	
	//		      		printf("bidx = %d and pidx = %d\n", bidx, pidx);	
					pidx++;
					bidx++;			
} // i1
} // i2
} // i3
}); // tbb
} // ii1
} // ii2
} // ii3
} // copy_blocked

void cpuProp::prop_naive(float *p0, const float *p1, const float *vel){ 
	tbb::tick_count t0 = tbb::tick_count::now();

	int count = 0;	
	int min = (1200 * 1200 * 672);
	int max = 0;
	float total_velocity = 0;
	tbb::parallel_for(tbb::blocked_range<int>(4,_nz-4),[&](
  	const tbb::blocked_range<int>&r){
  	for(int  i3=r.begin(); i3!=r.end(); ++i3){
//	for(int i3=4; i3 < _nz-4; i3++){
		for(int i2=4; i2 < _ny-4; i2++) {
			int ii=i2*_nx+4+_n12*i3;
			for(int i1=4; i1 < _nx-4; i1++,ii++) {
		       float x=
				p0[ii]=vel[ii]*
					      (
					coeffs[C0]*p1[ii]
					+coeffs[CX1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[CX2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[CX3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[CX4]*(p1[ii-4]+p1[ii+4])+
					+coeffs[CY1]*(p1[ii-_nx]+p1[ii+_nx])+
					+coeffs[CY2]*(p1[ii-2*_nx]+p1[ii+2*_nx])+
					+coeffs[CY3]*(p1[ii-3*_nx]+p1[ii+3*_nx])+
					+coeffs[CY4]*(p1[ii-4*_nx]+p1[ii+4*_nx])+
					+coeffs[CZ1]*(p1[ii-1*_n12]+p1[ii+1*_n12])+
					+coeffs[CZ2]*(p1[ii-2*_n12]+p1[ii+2*_n12])+
					+coeffs[CZ3]*(p1[ii-3*_n12]+p1[ii+3*_n12])+
					+coeffs[CZ4]*(p1[ii-4*_n12]+p1[ii+4*_n12])
				        )
				        +p1[ii]+p1[ii]-p0[ii];
			
				if (x != 0.0){
					//printf("%d ",ii);
					total_velocity+=vel[ii];
					count++; 
					if (ii > max)
						max = ii;
					if (ii < min)
						min = ii;
				}	      
			}
		}
	}
	});
	tbb::tick_count t1 = tbb::tick_count::now();
	std::cout << "The time in seconds for the prop_naive function is " << (t1-t0).seconds() << std::endl;
	printf("The total number of cells with a value are %d and the average velociy is %f\n",count, total_velocity/count);
	printf("The minimum cell number is %d and the maximum cell number is %d \n",min, max);
}

void cpuProp::prop_blocked(float* p0, const float* p1, float *vel){  
	tbb::tick_count t0 = tbb::tick_count::now();	

	int stride = STRIDE;
	int offset = 4;
	
	//calucalate dimension of new block size
	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);
	int blk=0;
	int b_nx = stride + 2 * offset;
	int b_n12 = (stride + 2 * offset) * (stride + 2 * offset);

	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
		for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
			int blk_ct = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
		//	for (int i3 = ii3; i3 < std::min(ii3+stride,_nz - offset); i3++){
			     int z_ct = i3 - ii3 + offset;
			     int y_ct = offset;
				for(int i2=ii2; i2 < std::min(ii2+stride, _ny - offset); i2++, y_ct++) {
				    int idx = i3 * _n12 + i2*_nx + ii1;
			    	    int ii  = blk_ct + y_ct * b_nx + z_ct * b_n12 + offset;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx - offset); i1++, ii++, idx++) {
					
					p0[idx]=vel[idx]*
					      (
					coeffs[C0]*p1[ii]
					+coeffs[CX1]*(p1[ii-1]+p1[ii+1])+
					+coeffs[CX2]*(p1[ii-2]+p1[ii+2])+
					+coeffs[CX3]*(p1[ii-3]+p1[ii+3])+
					+coeffs[CX4]*(p1[ii-4]+p1[ii+4])+

					+coeffs[CY1]*(p1[ii-b_nx]+p1[ii+b_nx])+
					+coeffs[CY2]*(p1[ii-2*b_nx]+p1[ii+2*b_nx])+
					+coeffs[CY3]*(p1[ii-3*b_nx]+p1[ii+3*b_nx])+
					+coeffs[CY4]*(p1[ii-4*b_nx]+p1[ii+4*b_nx])+
					
					+coeffs[CZ1]*(p1[ii-1*b_n12]+p1[ii+1*b_n12])+
					+coeffs[CZ2]*(p1[ii-2*b_n12]+p1[ii+2*b_n12])+
					+coeffs[CZ3]*(p1[ii-3*b_n12]+p1[ii+3*b_n12])+
					+coeffs[CZ4]*(p1[ii-4*b_n12]+p1[ii+4*b_n12])
				        )
				        +p1[ii]+p1[ii]-p0[idx];
					//printf("p0[%d] = %f\n", idx, p0[idx]); 
	} //i1
	} //i2
	} //i3
	}); //tbb
	} //ii1
	} //ii2
	} //ii3
	tbb::tick_count t1 = tbb::tick_count::now();
	std::cout << "The time in seconds for the prop_blocked function is " << (t1-t0).seconds() << std::endl;

} //prop_blocked


void cpuProp::prop_vec(float *p0, const float *p1, float *vel){
	tbb::tick_count t0 = tbb::tick_count::now();	

	int stride = STRIDE;
	int offset = 4;
	

	int blk_sz = (stride + 2*offset) * (stride + 2*offset) * (stride + 2*offset);
	int blk=0;
	int b_nx = stride + 2 * offset;
	int b_n12 = (stride + 2 * offset) * (stride + 2 * offset);

	 for(int ii3=offset; ii3 < _nz - offset ; ii3+=stride) {
		for(int ii2=offset; ii2 < _ny - offset; ii2+=stride) {
		   for (int ii1 = offset; ii1 < _nx - offset; ii1+=stride, blk++){
	//		printf("Now at block %d\n",blk);
			int blk_ct = blk*blk_sz;
			tbb::parallel_for(tbb::blocked_range<int>(ii3,std::min(ii3+stride,_nz)),
			[&] (const tbb::blocked_range<int>&r){
			for (int i3 = r.begin(); i3 != r.end(); ++i3){
	//		for (int i3 = ii3; i3 < std::min(ii3+stride,_nz - offset); i3++){
			     int z_ct = i3 - ii3 + offset;
			     int y_ct = offset;
				for(int i2=ii2; i2 < std::min(ii2+stride, _ny - offset); i2++, y_ct++) {
				    int idx = i3 * _n12 + i2*_nx + ii1;
			    	    int ii  = blk_ct + y_ct * b_nx + z_ct * b_n12 + offset;
				    for(int i1=ii1; i1 < std::min(ii1+stride, _nx - offset); i1+=AVX_SIMD_LENGTH) {
				
				//Broadcast coefficients
				__m512 _coeff0 = _mm512_set1_ps(coeffs[C0]);
				__m512 _coeff1 = _mm512_set1_ps(coeffs[CX1]);
				__m512 _coeff2 = _mm512_set1_ps(coeffs[CX2]);
				__m512 _coeff3 = _mm512_set1_ps(coeffs[CX3]);
				__m512 _coeff4 = _mm512_set1_ps(coeffs[CX4]);
				__m512 _coeff5 = _mm512_set1_ps(coeffs[CY1]);
				__m512 _coeff6 = _mm512_set1_ps(coeffs[CY2]);
				__m512 _coeff7 = _mm512_set1_ps(coeffs[CY3]);
				__m512 _coeff8 = _mm512_set1_ps(coeffs[CY4]);
				__m512 _coeff9 = _mm512_set1_ps(coeffs[CZ1]);
				__m512 _coeff10 = _mm512_set1_ps(coeffs[CZ2]);
				__m512 _coeff11 = _mm512_set1_ps(coeffs[CZ3]);
				__m512 _coeff12 = _mm512_set1_ps(coeffs[CZ4]);
				
				__m512 _o, _p0, _p1, _vel, _center;
				__m512 _xm1, _xm2, _xm3, _xm4, _xp1, _xp2, _xp3, _xp4;
				__m512 _ym1, _ym2, _ym3, _ym4, _yp1, _yp2, _yp3, _yp4;
				__m512 _zm1, _zm2, _zm3, _zm4, _zp1, _zp2, _zp3, _zp4;


				//load center values
				_p1 = _mm512_load_ps( p1 + idx);
				_p0 = _mm512_load_ps( p0 + idx);
				_vel = _mm512_load_ps( vel + idx); 
				//_center = _p1;		
				
				//x values
				_xm1 = _mm512_loadu_ps( p1 + idx - 1);
				_xm2 = _mm512_loadu_ps( p1 + idx - 2);
				_xm3 = _mm512_loadu_ps( p1 + idx - 3);
				_xm4 = _mm512_loadu_ps( p1 + idx - 4);
				_xp1 = _mm512_loadu_ps( p1 + idx + 1);
				_xp2 = _mm512_loadu_ps( p1 + idx + 2);
				_xp3 = _mm512_loadu_ps( p1 + idx + 3);
				_xp4 = _mm512_loadu_ps( p1 + idx + 4);

				//y values
				_ym1 = _mm512_loadu_ps( p1 + idx - (1 * b_nx));
				_ym2 = _mm512_loadu_ps( p1 + idx - (2 * b_nx));
				_ym3 = _mm512_loadu_ps( p1 + idx - (3 * b_nx));
				_ym4 = _mm512_loadu_ps( p1 + idx - (4 * b_nx));
				_yp1 = _mm512_loadu_ps( p1 + idx + (1 * b_nx));
				_yp2 = _mm512_loadu_ps( p1 + idx + (2 * b_nx));
				_yp3 = _mm512_loadu_ps( p1 + idx + (3 * b_nx));
				_yp4 = _mm512_loadu_ps( p1 + idx + (4 * b_nx));

				//z values
				_zm1 = _mm512_loadu_ps( p1 + idx - 1 * b_n12);
				_zm2 = _mm512_loadu_ps( p1 + idx - 2 * b_n12);
				_zm3 = _mm512_loadu_ps( p1 + idx - 3 * b_n12);
				_zm4 = _mm512_loadu_ps( p1 + idx - 4 * b_n12);
				_zp1 = _mm512_loadu_ps( p1 + idx + 1 * b_n12);
				_zp2 = _mm512_loadu_ps( p1 + idx + 2 * b_n12);
				_zp3 = _mm512_loadu_ps( p1 + idx + 3 * b_n12);
				_zp4 = _mm512_loadu_ps( p1 + idx + 4 * b_n12);

				//calculations:
				_o = _mm512_setzero_ps();
				_o = _mm512_fmadd_ps(_coeff0, _p1, _o);
				//coeffs[CX1] * (p1[ii-1] + p[ii+i])
				_o = _mm512_fmadd_ps(_coeff1, _xm1, _o);
				_o = _mm512_fmadd_ps(_coeff1, _xp1, _o);
				//coeffs[CX2] * (p1[ii-2] + p[ii+2])
				_o = _mm512_fmadd_ps(_coeff2, _xm2, _o);
				_o = _mm512_fmadd_ps(_coeff2, _xp2, _o);
				//coeffs[CX3] * (p1[ii-3] + p[ii+3])
				_o = _mm512_fmadd_ps(_coeff3, _xm3, _o);
				_o = _mm512_fmadd_ps(_coeff3, _xp3, _o);
				//coeffs[CX4] * (p1[ii-4] + p[ii+4])
				_o = _mm512_fmadd_ps(_coeff4, _xm4, _o);
				_o = _mm512_fmadd_ps(_coeff4, _xp4, _o);
				
				//coeffs[CY1] * (p1[ii-_nx] + p[ii+_nx])
				_o = _mm512_fmadd_ps(_coeff5, _ym1, _o);
				_o = _mm512_fmadd_ps(_coeff5, _yp1, _o);
				//coeffs[CY2] * (p1[ii-2*_nx] + p[ii+2*_nx])
				_o = _mm512_fmadd_ps(_coeff6, _ym2, _o);
				_o = _mm512_fmadd_ps(_coeff6, _yp2, _o);
				//coeffs[CY3] * (p1[ii-3*_nx] + p[ii+3*_nx])
				_o = _mm512_fmadd_ps(_coeff7, _ym3, _o);
				_o = _mm512_fmadd_ps(_coeff7, _yp3, _o);
				//coeffs[CY4] * (p1[ii-4*_nx] + p[ii+4*_nx])
				_o = _mm512_fmadd_ps(_coeff8, _ym4, _o);
				_o = _mm512_fmadd_ps(_coeff8, _yp4, _o);
				
				//coeffs[CZ1] * (p1[ii-_n12] + p[ii+_n12])
				_o = _mm512_fmadd_ps(_coeff9, _zm1, _o);
				_o = _mm512_fmadd_ps(_coeff9, _zp1, _o);
				//coeffs[CZ2] * (p1[ii-_2*n12] + p[ii+_2*n12])
				_o = _mm512_fmadd_ps(_coeff10, _zm2, _o);
				_o = _mm512_fmadd_ps(_coeff10, _zp2, _o);
				//coeffs[CZ3] * (p1[ii-3*_n12] + p[ii+3*_n12])
				_o = _mm512_fmadd_ps(_coeff11, _zm3, _o);
				_o = _mm512_fmadd_ps(_coeff11, _zp3, _o);
				//coeffs[CZ4] * (p1[ii-4*_n12] + p[ii+4*_n12])
				_o = _mm512_fmadd_ps(_coeff12, _zm4, _o);
				_o = _mm512_fmadd_ps(_coeff12, _zp4, _o);

				_o = _mm512_mul_ps(_vel, _o);
				_o = _mm512_add_ps(_p1, _o);
				_o = _mm512_add_ps(_p1, _o);
				_o = _mm512_sub_ps(_o, _p0);
				//write final value		
				_mm512_stream_ps(p0 + idx, _o);
				idx+=AVX_SIMD_LENGTH;
				ii+=AVX_SIMD_LENGTH;
	} //i1
	} //i2
	} //i3
	}); //tbb
	} // ii1
	} // ii2
	} // ii3
	tbb::tick_count t1 = tbb::tick_count::now();
	std::cout << "The time in seconds for the prop_vec function is " << (t1-t0).seconds() << std::endl;
	//profiler.stop();
} //prop_vector


void cpuProp::transferSincTableD(int nsinc, int jtd, std::vector<std::vector<float>> &table){
// transfer_sinc_table_d(nsinc,jtd,myr.table);
	_nsincD=nsinc;
	_jtdD=jtd;
	_tableD=table;

}
void cpuProp::transferSourceFunc(int npts,int nt_big,std::vector<int> &locs,float *vals){
	_nptsS=npts; _ntSrc=nt_big; _locsS=locs; _sourceV=vals;
	
}
void cpuProp::transferVelFunc1(int nx, int ny, int nz, float *vloc){
	_nx=nx; _ny=ny; _nz=nz; _vel1=vloc;
}
void cpuProp::transferVelFunc2(int nx, int ny, int nz, float *vloc){
	_nx=nx; _ny=ny; _nz=nz; _vel2=vloc;

}

void cpuProp::transferReceiverFunc(int nx, int ny, int nt, std::vector<int> &locs,
	float * rec){
	_nRecs=nx*ny; _ntRec=nt; _locsR=locs; _rec=rec;
	int ii=0;
	
}
void cpuProp::transferSincTableS(int nsinc, int jts, std::vector<std::vector<float>> &table){
  _tableS=table;
	_nsincS=nsinc;
	_jtsS=jts;
//	_tableS=table;
}
#define C_C00(d) (8.0/(5.0*(d)*(d)))

void cpuProp::createSpace(float d1, float d2, float d3,float bc_a, float bc_b, float bc_y,
	int nx, int ny, int nz){
	_bcA=bc_a; _bcB=bc_b; bc_y=_bcY;
	_nx=nx; _ny=ny; _nz=nz;
	_n12=_nx*_ny;
	coeffs.resize(13);
	coeffs[0]=-8/12.5/12.5;
	coeffs[1]=coeffs[2]=coeffs[3]=1./12.5/12.5;
	coeffs[4]=coeffs[5]=coeffs[6]=0.;
	coeffs[7]=coeffs[8]=coeffs[9]=0.;
	coeffs[10]=coeffs[11]=coeffs[12]=0.;

	
	coeffs[0]=-1025.0/576.0*(C_C00(d1)+C_C00(d2)+C_C00(d3));
	coeffs[1]=C_C00(d1);
	coeffs[2]=C_C00(d2);
	coeffs[3]=C_C00(d3);
	coeffs[4]=-C_C00(d1)/8.0;
	coeffs[5]=-C_C00(d2)/8.0;
	coeffs[6]=-C_C00(d3)/8.0;
	coeffs[7]=C_C00(d1)/63.0;
	coeffs[8]=C_C00(d2)/63.0;
	coeffs[9]=C_C00(d3)/63.0;
	coeffs[10]=-C_C00(d1)/896.0;
	coeffs[11]=-C_C00(d2)/896.0;
	coeffs[12]=-C_C00(d3)/896.0;
	
	
	

	_n123=_n12*nz;
	_bcB=.0005;
	_bcA=40;
	_bound.resize(40);
	for(int i=0;i < _bound.size(); i++) _bound[i]=expf(-_bcB*(_bcA-i));
/*
	coeffs[1]=.8/d1/d1;
	coeffs[2]=.8/d2/d2;
	coeffs[3]=.8/d3/d3;
	coeffs[4]=-.1/d1/d1;
	coeffs[5]=-.1/d2/d2;
	coeffs[6]=-.1/d3/d3;
    coeffs[7]=.0126984126/d1/d1;
    coeffs[8]=.0126984126/d2/d2;
    coeffs[9]=.0126984126/d3/d3;
    coeffs[10]=-.0008928571428/d1/d1;
    coeffs[11]=-.0008928571428/d2/d2;
    coeffs[12]=-.0008928571428/d3/d3;
    coeffs[0]=0;
    for(int i=1; i < 13; i++) coeffs[0]-=coeffs[i];
    */



	//create_gpu_space(d1,d2,d3,bc_a,bc_b,bc_b_y,nx,ny,nz);
}
