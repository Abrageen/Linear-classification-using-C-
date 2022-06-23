#include <stdio.h>
#include <stdlib.h>
#include<math.h>

double* initialize(int l)
{
    double* w = (double*) malloc(l * sizeof(double));
    for (int i=0 ; i<l ; i++)
    {
        w[i]=0.5;
    }
    return w;
}
double sigmoid (double z)
{
    double s;
    z=(-z);
    s = 1/(1+exp(z));
    return s;
}

void forward_propagate(double w[], double b, double X[], double Y[], int l, double* cost)
{
    double* A = (double*) malloc(l * sizeof(double));
    double sum;
    float m=l;
    for (int i=0 ; i<l ; i++)
    {
        A[i]=sigmoid((w[i]*X[i])+b);
        sum+=Y[i]*log(A[i])+ (1-Y[i])*log(1-A[i]);
    }
    *cost=((-1)/m)*sum;
}

double* backward_propagate(double w[], double b, double X[], double Y[], int l, double* db)
{
    float m=l;
    double sumb;
    double* A = (double*) malloc(l * sizeof(double));
    double* dw = (double*) malloc(l * sizeof(double));
    for (int i=0 ; i<l ; i++)
    {
        A[i]=sigmoid((w[i]*X[i])+b);
        dw[i] = (1/m)* (X[i]*(A[i]-Y[i]));
        sumb = A[i]-Y[i];
    }
    *db=(1/m)*sumb;
    return dw;
}
double* optimise(double w[], double* b, double X[], double Y[], int l, int iterations, double learning_rate)
{
    double db,cost,b_T;
    double* dw = (double*) malloc(l * sizeof(double));
    for (int i=0 ; i<iterations ; i++)
    {
        forward_propagate(w,*b,X,Y,l,&cost);
        dw=backward_propagate(w,*b,X,Y,l,&db);
        for (int j=0 ; j<l ; j++)
        {
            w[j]=w[j]-(learning_rate*dw[j]);
        }
        b_T=-(learning_rate*db);
        if(i%10==0)
        {
            printf("%f\n\n",cost);
        }
    }
    *b=b_T;
    return w;
}

int main(int argc, const char ** argv[])
{
    double X[3]={1.1,2.2,3.3},Y[3]={0,1,0};
    double b=0.5;
    int l=3;
    double* w=initialize(l);
    w=optimise(w,&b,X,Y,l,100,0.0001);
    return (0);
}
