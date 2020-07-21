/* The MathWorks Inc. 2019*/ 

/* ResNet50 demo main.cu file with OpenCV interfaces to read and display data. */

#include "resnet50_wrapper.h"
#include "main_resnet50.h"
#include "resnet50_wrapper_terminate.h"
#include "resnet50_wrapper_initialize.h"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <stdlib.h>

#define IMG_WIDTH 224
#define IMG_HEIGHT 224
#define IMG_CH 3
#define VID_DEV_ID -1
        
using namespace cv;
using namespace std;
       
static void main_resnet50_wrapper();

/* 
* Convert BGR data to RGB data, without this conversion the predictions 
* will be bad
*/
static void argInit_224x224x3_real32_T(real32_T *input, Mat & im)
{
    for(int j=0;j<224*224;j++)
    {
        //BGR to RGB
        input[2*224*224+j]=(float)(im.data[j*3+0]);
        input[1*224*224+j]=(float)(im.data[j*3+1]);
        input[0*224*224+j]=(float)(im.data[j*3+2]);
    }
}

int cmpfunc(const void * a, const void * b, void * r)
{
	float x =  ((float*)r)[*(int*)b] - ((float*)r)[*(int*)a] ;
	return ( x > 0 ? ceil(x) : floor(x) );
}

void top( float* r, int* top5 )
{
    int t[1000];
    for(int i=0; i<1000; i++)
        t[i]=i;
    qsort_r(t, 1000, sizeof(int), cmpfunc, r);
    top5[0]=t[0];
    top5[1]=t[1];
    top5[2]=t[2];
    top5[3]=t[3];
    top5[4]=t[4];
    return;
}

/* Write the prediction scores on the output video frame */
void writeData(float *output,  char synsetWords[1000][100], Mat & frame, float fps)
{
  int top5[5];
  top(output, top5);

  copyMakeBorder(frame, frame, 0, 0, 400, 0, BORDER_CONSTANT, CV_RGB(0,0,0));
  char strbuf[50];
  sprintf (strbuf, "%.2f FPS", fps);
  putText(frame, strbuf, Point(30,30), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);
  sprintf(strbuf, "%4.1f%% %s", output[top5[0]]*100, synsetWords[top5[0]]);
  putText(frame, strbuf, Point(30,80), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);
  sprintf(strbuf, "%4.1f%% %s", output[top5[1]]*100, synsetWords[top5[1]]);
  putText(frame, strbuf, Point(30,130), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);
  sprintf(strbuf, "%4.1f%% %s", output[top5[2]]*100, synsetWords[top5[2]]);
  putText(frame, strbuf, Point(30,180), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);
  sprintf(strbuf, "%4.1f%% %s", output[top5[3]]*100, synsetWords[top5[3]]);
  putText(frame, strbuf, Point(30,230), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);
  sprintf(strbuf, "%4.1f%% %s", output[top5[4]]*100, synsetWords[top5[4]]);
  putText(frame, strbuf, Point(30,280), FONT_HERSHEY_DUPLEX , 1.0, CV_RGB(220,220,220), 1);

  imshow("resnet Demo", frame);
}

/* Read the class lables from the .txt file*/
int prepareSynset(char synsets[1000][100])
{
  FILE* fp1 = fopen("synsetWords_resnet50.txt", "r");
  if (fp1 == 0) return -1;
  for(int i=0; i<1000; i++)
  {
    fgets(synsets[i], 100, fp1);
    strtok(synsets[i], "\n");
  }
  fclose(fp1);
  return 0;
}

static void main_resnet50_wrapper(void)
{
  real32_T out[1000];
  static real32_T b[150528];

  char synsetWords[1000][100];
  if (prepareSynset(synsetWords) == -1)
  {
    printf("ERROR: Unable to find synsetWords_resnet50.txt\n");
    exit(0);
  }   

  Mat oFrame, cFrame;
  /* Initialize function 'resnet50_wrapper' input arguments. */
  /* Initialize function input argument 'in'. */
  /* Call the entry-point 'resnet50_wrapper'. */

  /* Create a Video capture object */
  VideoCapture cap(VID_DEV_ID);
  if(!cap.isOpened())
  {
    cout << "can't open camera" << endl;
    exit(0);
  }
  namedWindow("resnet Demo",WINDOW_NORMAL );
  resizeWindow("resnet Demo", 1000,1000);
  float fps=0;	
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);		
  
  while(1)
  {    
    cap >> oFrame;
    resize(oFrame,cFrame,Size(IMG_WIDTH,IMG_HEIGHT));
    
    /* convert from BGR to RGB*/
    argInit_224x224x3_real32_T(b,cFrame);
    cudaEventRecord(start);
    
    /* call the resent predict  function*/
    resnet50_wrapper(b, out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = -1.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fps = fps*.9+1000.0/milliseconds*.1;

    /* Write the prediction on the output frame */
    writeData(out, synsetWords, oFrame, fps);
    if(waitKey(1)%256 == 27 ) break; // stop when ESC key is pressed
  }
  
}

int32_T main(int32_T argc, const char * const argv[])
{
  (void)argc;
  (void)argv;
  
  /* Call the application intialize function */
  resnet50_wrapper_initialize();
  
  /* Call the resnet predict function */
  main_resnet50_wrapper();

 /* Call the application terminate function */
  resnet50_wrapper_terminate();
  return 0;
}

