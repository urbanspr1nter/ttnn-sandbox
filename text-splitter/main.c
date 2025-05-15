#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

#define SPLIT_PERCENTAGE 0.85

int main(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: text-splitter sourceFilePath destTrainDataFile destValidationDataFile \n");
    exit(1);
  }

  char* filename = argv[1];
  char* destTrainDataFile = argv[2];
  char* destValDataFile = argv[3];
  
  struct stat fileStat;
  if (stat(filename, &fileStat) == -1) {
    fprintf(stderr, "Could not get the statistics for the file: %s\n", filename);
    exit(1);
  }

  uint64_t allocationSize = sizeof(char) * fileStat.st_size + 1;
  char* buffer = malloc(allocationSize);

  FILE* f = fopen(filename, "r");
  fread(buffer, sizeof(char), fileStat.st_size, f);

  uint64_t splitIdx = (uint64_t)((float)SPLIT_PERCENTAGE * fileStat.st_size);
  printf("Number of characters total: %ld, Split index: %ld\n", fileStat.st_size, splitIdx);

  uint64_t lenForTrain = fileStat.st_size - (fileStat.st_size - splitIdx) + 1;
  uint64_t lenForValidation = (fileStat.st_size - splitIdx) + 1;

  char* trainData = malloc(sizeof(char) * lenForTrain);
  char* valData = malloc(sizeof(char) * lenForValidation);

  uint64_t i = 0;
  uint64_t j = 0;
  while (i < lenForTrain) {
    trainData[i] = buffer[j];
    i++;
    j++;
  }
  trainData[i] = '\0';

  FILE* tf = fopen(destTrainDataFile, "w");
  fwrite(trainData, sizeof(char), lenForTrain, tf);
  fclose(tf);

  i = 0;
  while (i < lenForValidation) {
    valData[i] = buffer[j];
    i++;
    j++;
  }
  valData[i] = '\0';

  FILE* vf = fopen(destValDataFile, "w");
  fwrite(valData, sizeof(char), lenForValidation, vf);
  fclose(vf);

  fclose(f);

  return 0;
}