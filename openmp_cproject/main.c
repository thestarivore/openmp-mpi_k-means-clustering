#include <stdlib.h>
#include <stdio.h>

int main() {
    FILE *myFile;
    float x=0.0f, y=0.0f;
    char c1,c2;
    char datasetFile[] = "../dataset_display/dataset.csv";

    //Get number of rows of the dataset
    int n_rows = countlines(datasetFile);
    printf("Number of rows in the file: %d\n", n_rows);

    //Open dataset file
    myFile = fopen(datasetFile, "r");

    //Set Array
    float * dataset = (float*) malloc(2 * n_rows * sizeof(float));

    //Read the Header first
    fscanf(myFile, "%c,%c ", &c1, &c2);
    if(c1=='X' && c2=='Y'){
        //Read file and store it into the array
        int i=0;
        while (fscanf(myFile, "%f,%f ", &x, &y) != EOF){
            //printf("Number is: %f, %f\n\n", x, y);
            dataset[0][i]=x;
            dataset[1][i]=y;
        }
    }

    fclose(myFile);
    system("python3 ../dataset_display/main.py");
}


int countlines(char *filename)
{
    // count the number of lines in the file called filename
    FILE *fp = fopen(filename,"r");
    int ch=0;
    int lines=0;

    if (fp == NULL)
        return 0;

    while(!feof(fp))
    {
      ch = fgetc(fp);
      if(ch == '\n')
      {
        lines++;
      }
    }
    fclose(fp);
    return lines;
}
