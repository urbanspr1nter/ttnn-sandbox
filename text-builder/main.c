#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <assert.h>

struct DirectoryStats {
  uint64_t totalSize;
  uint64_t totalFiles;
};

/**
 * Gets the file size of a file.
 */
uint64_t getFileSize(char* path) {
  struct stat fileStats;

  if (stat(path, &fileStats) == -1) {
    fprintf(stderr, "Couldn't open file to get file size: %s\n", path);
    exit(1);
  }

  return fileStats.st_size;
}

/**
 * Gets some basic stats about the directory: total size and total files excluding . and ..
 */
struct DirectoryStats* getTotalSizeOfFilesFromDirectory(char* baseDirPath) {
  DIR* dir;
  struct dirent* entry;

  dir = opendir(baseDirPath);
  if (dir == NULL) {
    fprintf(stderr, "Error opening the directory: %s\n", baseDirPath);
    exit(1);
  }

  uint64_t totalSize = 0;
  uint64_t totalFiles = 0;

  while ((entry = readdir(dir)) != NULL) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    char* fullpath = malloc(sizeof(char) * ((strlen(baseDirPath) + strlen(entry->d_name) + 2)));
    sprintf(fullpath, "%s/%s", baseDirPath, entry->d_name);
    fullpath[strlen(baseDirPath) + strlen(entry->d_name) + 1] = '\0';

    FILE* f = fopen(fullpath, "r");
    if (f == NULL) {
      fprintf(stderr, "Invalid file at: %s\n", fullpath);
      exit(1);
    }
    
    totalSize += getFileSize(fullpath);
    totalFiles++;

    fclose(f);
    free(fullpath);
  }

  closedir(dir);

  struct DirectoryStats* result = malloc(sizeof(struct DirectoryStats));
  result->totalSize = totalSize;
  result->totalFiles = totalFiles;

  printf("Total files: %ld\nTotal size of files: %ld bytes\n", result->totalFiles, result->totalSize);

  return result;
}

char* buildStringFromFilesInDirectory(char* baseDirPath, struct DirectoryStats* dirStats) {
  // consider the new line characters to separate
  uint64_t allocationSize = dirStats->totalSize + (sizeof(char) *  (2* dirStats->totalFiles)) + 1;
  char* result = calloc(allocationSize, sizeof(char));

  DIR* dir;
  struct dirent* entry;

  dir = opendir(baseDirPath);
  if (dir == NULL) {
      fprintf(stderr, "Error opening the direcotry: %s\n", baseDirPath);
      exit(1);
  }

  uint64_t idx = 0;
  uint64_t currIdx = 0;
  while ((entry = readdir(dir)) != NULL) {

    // Skip the special current and parent directory pointers.
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    char* fullpath = malloc(sizeof(char) * ((strlen(baseDirPath) + strlen(entry->d_name) + 2)));
    sprintf(fullpath, "%s/%s", baseDirPath, entry->d_name);

    fullpath[strlen(baseDirPath) + strlen(entry->d_name) + 1] = '\0';
    
    uint64_t fileSize = getFileSize(fullpath);

    FILE* f = fopen(fullpath, "r");

    char* buffer = malloc(sizeof(char) * fileSize + 1);
    fread(buffer, sizeof(char), fileSize + 1, f);

    int i = 0;
    while (i < fileSize) {
      result[currIdx] = buffer[i];
      currIdx++;
      i++;
    }
    result[currIdx] = '\n';
    currIdx++;

    fclose(f);
    free(buffer);
    free(fullpath);

    idx++;
  }
  result[currIdx] = '\0';

  closedir(dir);

  printf("Total file size read into memory: %ld, Number of additional characters for new line: %ld\n", currIdx, currIdx - dirStats->totalSize);

  assert(dirStats->totalSize == (currIdx - dirStats->totalFiles)); 
  return result;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: text-builder base_data_dir output_file\n");
    exit(1);
  } 

  char* basedirPath = argv[1];
  char* outputFilePath = argv[2];

  struct DirectoryStats* dirStats = getTotalSizeOfFilesFromDirectory(basedirPath);
  char* result = buildStringFromFilesInDirectory(basedirPath, dirStats);

  FILE* f = fopen(outputFilePath, "w");
  fwrite(result, sizeof(char), strlen(result), f);

  fclose(f);
  free(result);
  free(dirStats);

  return 0;
}
