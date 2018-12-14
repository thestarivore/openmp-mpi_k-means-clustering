#include <stdlib.h>
#include <stdio.h>



int main() {
    FILE *myFile;
    myFile = fopen("data1.cvs", "r");

    //read file into array
    int noUse;
    float x=0.0f, y=0.0f;

    while (fscanf(myFile, "%d %f %f ", &noUse, &x, &y) != EOF){
        printf("Number is: %f, %f\n\n", x, y);
    }

    
}


