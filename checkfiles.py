# check the amount of files in a folder

import os

def count_files_in_folder(folder):
    return len(os.listdir(folder))

if __name__ == "__main__":
    #make windows path readable for linux
    folder = "./hackupc-2024/embeddings"
   
    print(count_files_in_folder(folder))