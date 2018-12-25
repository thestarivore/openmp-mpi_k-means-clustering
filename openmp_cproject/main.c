#include <stdlib.h>
#include <stdio.h>

int main() {
    FILE *myFile;
    myFile = fopen("../dataset_display/dataset.csv", "r");

    //read file into array
    float x=0.0f, y=0.0f;
    char c1,c2;

    //Read the Header first
    fscanf(myFile, "%c,%c ", &c1, &c2);
    if(c1=='X' && c2=='Y'){
        while (fscanf(myFile, "%f,%f ", &x, &y) != EOF){
            printf("Number is: %f, %f\n\n", x, y);
        }
    }

    system("python3 ../dataset_display/main.py");
}

