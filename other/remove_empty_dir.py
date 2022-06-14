import os


def main():
    folder_path: str = 'D:/viterbi linkage/dataset/'

    save_dir = folder_path + 'save_directory_enhancement/'

    dir_list = os.listdir(save_dir)

    for dir in dir_list:
        if os.path.isdir(save_dir + dir):
            subdir = save_dir + dir
            is_dir_empty: bool = len(os.listdir(subdir)) == 0

            if is_dir_empty:
                print("remove: ", dir)
                os.rmdir(subdir)


if __name__ == '__main__':
    main()