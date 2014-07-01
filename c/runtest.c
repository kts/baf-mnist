/*

Todo:
- needs more error checking.

Note:
 This does matrix multiplication (and 
 other operations) directly.
 For any real work, you'd want to use 
 an optimized library (or hardware-accelerated)
 for this.  Just an example...

*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libbaf.h>


/*
  Compute argmax(Wx+b)

  inputs,
   W: (Nd,Nc)  weights
   b: (Nc,)    biases
   X: (N,Nd)   data

  where,
   Nd:num dimensions
   N: num datapoints
   Nc:num classes

  output:
  Yhat (N,) : hypothesis labels.

*/
int argmax_layer(float** W,
		 float* b,
		 float** X,
		 int N,
		 int Nd,
		 int Nc,
		 uint8_t* Yhat
		 ) {
  int i,j;
  int k;

  float *w;
  float xj;
  float* out = malloc(sizeof(float) * Nc);

  float *x; // one vector

  for (k=0; k<N; k++) {
    
    x = X[k];

    // start with bias:
    for (i=0; i<Nc; i++)
      out[i] = b[i];

    for (j=0; j<Nd; j++) {
      w  = W[j];
      xj = x[j];
      
      for (i=0; i<Nc; i++) {
	out[i] += *w++  *  xj;
      }
    } // dims

    int   maxarg = 0;
    float maxval = out[0];

    for (i=1; i<Nc; i++) {
      if (out[i] > maxval) {
	maxval = out[i];
	maxarg = i;
      }
    }

    Yhat[k] = maxarg;

  }

  free(out);

  return 0;
}


int main(int argc, char **argv) {
  /*

usage: modelfile
--modelfile should contain 'W', 'b'
--datafile should ...

   */
  char *modelfile = "model.baf";
  char *datafile  = "/tmp/mnist.baf";

  char *sel_w = "W";
  char *sel_b = "b";

  char *sel_x = "Xs";
  char *sel_y = "Ys";

  /////
 
  float **W;
  float *b;

  printf("Reading file '%s'...\n", modelfile);
  int res;

  int N, dim, tmpint;
  int num_classes; // 10 for mnist.

  res = baf_read_matrix_f32(modelfile, sel_w,
			    &W, &dim, &num_classes);
  printf("Read W: (%d,%d)\n", dim, num_classes);

  res = baf_read_vector_f32(modelfile,sel_b,
			    &b, &tmpint);
  printf("Read b: (%d,)\n", tmpint);

  float **X;
  uint8_t *Y;
  uint8_t *Yhat;

  int n1,n2;
  res = baf_read_matrix_f32(datafile,sel_x,&X, &N, &n2);
  printf("Read X: (%d,%d)\n", N, n2);

  //float *x = X[1];

  printf("N   = %d\n", N);
  printf("dim = %d\n", dim);

  Yhat = (uint8_t*)malloc(sizeof(uint8_t) * N);

  res = argmax_layer(W,b,X,
		     N,
		     dim,
		     10,
		     Yhat);
  int ny;
  int ii;

  res = baf_read_vector_uint8(datafile,sel_y,&Y,&ny);

  int nerr = 0;
  for (ii=0; ii<N; ii++) {
    if (Y[ii] != Yhat[ii])
      nerr++;
  }

  printf("Error: %.2f%% (%d / %d)\n", 
	 100.0 * (float)nerr / N,
	 nerr, N);

  free(Yhat);
  // free loaded...

  return 0;
}
