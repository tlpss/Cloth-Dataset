import pathlib
import os 
def create_filtered_symlink_dataset(root_dir_path: str, target_dir_for_symlinks: str, suffixes_to_symlink = [".png"]):
    """creates a "view" on the dataset by creating symlinks to the images with the appropriate suffixes in the dataset.
    This is useful for labeling: cvat cannot deal with non-image files and we also don't want to label right views manually for example."""
    # traverse all files in the root_dir
    # find all png's 
    # create symlinks to the png's in the target_dir_for_symlinks
    # if they have suffix zed.png, ...
    root_dir = pathlib.Path(root_dir_path)
    # check if root_dir exists and has the expected subfolders
    assert root_dir.exists()
    assert (root_dir / "train").exists()
    assert (root_dir / "test").exists()

    num_files = 0

    # traverse root_dir to find all files
    for root, dirs, files in os.walk(root_dir):
        relative_root_path = os.path.relpath(root, root_dir)
        symlink_dir = os.path.join(target_dir_for_symlinks, relative_root_path)
        os.makedirs(symlink_dir, exist_ok=True)
        print(root)
        for relative_file_path in files:
            for suffix in suffixes_to_symlink:
                if relative_file_path.endswith(suffix):
                    os.symlink(os.path.join(root_dir_path,relative_file_path), os.path.join(symlink_dir, relative_file_path))



    print(num_files)
    

if __name__ == "__main__":
    create_filtered_symlink_dataset("/home/tlips/Documents/cloth-dataset/Dataset","/home/tlips/Documents/cloth-dataset/FilteredDataset")

